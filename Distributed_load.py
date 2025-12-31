import socket
import struct
import time
import threading
import queue
import sys
import urllib.request
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt

SENSOR_IP = "192.168.1.1"
RDT_PORT = 49152

HDR = 0x1234
CMD_STOP = 0x0000
CMD_START_REALTIME = 0x0002
CMD_START_BUFFERED = 0x0003

def fetch_counts_factors(sensor_ip: str):
    url = f"http://{sensor_ip}/netftapi2.xml"
    with urllib.request.urlopen(url, timeout=2.0) as resp:
        xml_bytes = resp.read()
    root = ET.fromstring(xml_bytes)
    cfgcpf = int(root.findtext("cfgcpf"))
    cfgcpt = int(root.findtext("cfgcpt"))
    fu = root.findtext("scfgfu")
    tu = root.findtext("scfgtu")
    return cfgcpf, cfgcpt, fu, tu

def pack_rdt_request(cmd: int, sample_count: int = 0) -> bytes:
    return struct.pack("!HHI", HDR, cmd, sample_count)

def unpack_rdt_record(payload: bytes):
    if len(payload) != 36:
        raise ValueError(f"Unexpected payload length {len(payload)} (expected 36)")
    rdt_seq, ft_seq, status, fx, fy, fz, tx, ty, tz = struct.unpack("!IIIiiiiii", payload)
    return rdt_seq, ft_seq, status, (fx, fy, fz, tx, ty, tz)

def sensor_to_model_axes(Fx_s, Fy_s, Fz_s, Mx_s, My_s, Mz_s):
    Fx_m = Fz_s
    Fy_m = Fx_s
    Fz_m = Fy_s
    Mx_m = Mz_s
    My_m = Mx_s
    Mz_m = My_s
    return Fx_m, Fy_m, Fz_m, Mx_m, My_m, Mz_m

class PdfEBBeamModel:
    def __init__(self, needle_length_m=0.200, needle_diameter_m=1.27e-3,
                 E=200e9, beta_deg=30.0, c_m=4.8e-3, N_grid=151):
        self.L = needle_length_m
        self.d = needle_diameter_m
        self.E = E
        self.mu = 0.28
        self.A = np.pi * (self.d / 2.0) ** 2
        self.I = np.pi * (self.d ** 4) / 64.0
        self.beta = np.deg2rad(beta_deg)
        self.c = c_m
        self.N = int(N_grid) if N_grid >= 9 else 151
        self.x = np.linspace(0.0, self.L, self.N)
        self.dx = self.x[1] - self.x[0]
        self.f_fric = 0.0
        self.f_insert = 0.05
        self.e_out = self.L - self.f_insert
        self._last_dir = np.array([1.0, 0.0])

    def update_user_inputs(self, f_fric_N: float, f_insert_m: float):
        self.f_fric = max(0.0, float(f_fric_N))
        self.f_insert = min(max(0.0, float(f_insert_m)), self.L)
        self.e_out = self.L - self.f_insert

    def _build_distributed_load(self, w_max):
        w = np.zeros_like(self.x)
        if self.f_insert <= 1e-9 or w_max <= 0.0:
            return w
        e = self.e_out
        for i, xi in enumerate(self.x):
            if xi >= e:
                w[i] = w_max * (xi - e) / max(self.f_insert, 1e-12)
        return w

    def _solve_beam_deflection(self, w, Fnt):
        N = self.N
        dx = self.dx
        EI = self.E * self.I
        A = np.zeros((N, N))
        b = np.zeros(N)
        A[0, 0] = 1.0
        b[0] = 0.0
        A[1, 0] = -3.0 / (2.0 * dx)
        A[1, 1] =  4.0 / (2.0 * dx)
        A[1, 2] = -1.0 / (2.0 * dx)
        b[1] = 0.0
        for i in range(2, N-2):
            A[i, i-2] = 1.0 / (dx ** 4)
            A[i, i-1] = -4.0 / (dx ** 4)
            A[i, i]   = 6.0 / (dx ** 4)
            A[i, i+1] = -4.0 / (dx ** 4)
            A[i, i+2] = 1.0 / (dx ** 4)
            b[i] = w[i] / EI
        A[N-2, N-3] = 1.0 / (dx ** 2)
        A[N-2, N-2] = -2.0 / (dx ** 2)
        A[N-2, N-1] = 1.0 / (dx ** 2)
        b[N-2] = 0.0
        A[N-1, N-4] = -1.0 / (dx ** 3)
        A[N-1, N-3] =  3.0 / (dx ** 3)
        A[N-1, N-2] = -3.0 / (dx ** 3)
        A[N-1, N-1] =  1.0 / (dx ** 3)
        b[N-1] = Fnt / EI
        try:
            v = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            v = np.linalg.lstsq(A, b, rcond=None)[0]
        phi_tip = (3.0 * v[N-1] - 4.0 * v[N-2] + v[N-3]) / (2.0 * dx)
        v_tip = float(v[N-1])
        return v_tip, phi_tip

    def compute_deflection(self, Fx, Fy, Fz, Mx, My, Mz):
        Fm = np.hypot(Fx, Fy)
        Mm = np.hypot(Mx, My)
        Mr = -(Mm - Fm * self.c)
        F_long = Fz - self.f_fric
        Fnt = F_long / np.tan(self.beta) if abs(np.tan(self.beta)) > 1e-12 else 0.0
        if np.isnan(Fnt) or np.isinf(Fnt):
            Fnt = 0.0
        if self.f_insert > 1e-9:
            w_max = 2.0 * max(0.0, (Fm - Fnt)) / self.f_insert
        else:
            w_max = 0.0
        w = self._build_distributed_load(w_max)
        v_tip, phi_tip = self._solve_beam_deflection(w, max(0.0, Fnt))
        if Fm > 1e-12:
            vec = np.array([Fy, Fz], dtype=float)
            norm = np.linalg.norm(vec)
            if norm > 1e-12:
                dir_y, dir_z = vec / norm
            else:
                dir_y, dir_z = self._last_dir
        else:
            dir_y, dir_z = self._last_dir
        self._last_dir = np.array([dir_y, dir_z])
        Uy = v_tip * dir_y
        Uz = v_tip * dir_z
        Ux = (Fz * self.L) / (self.E * self.A)
        theta_z = phi_tip * dir_y
        theta_y = phi_tip * dir_z
        return {
            "Ux": Ux, "Uy": Uy, "Uz": Uz,
            "theta_y": theta_y, "theta_z": theta_z,
            "Fm": Fm, "Mm": Mm, "Mr": Mr,
            "Fnt": Fnt,
            "Uyz": np.hypot(Uy, Uz)
        }

class LiveProcessor:
    def __init__(self, needle_length_m=0.200, display_period_s=0.1, history_s=10.0):
        self.model = PdfEBBeamModel(needle_length_m=needle_length_m)
        self.display_period_s = display_period_s
        self.history_s = history_s
        self.q = queue.Queue(maxsize=3000)
        self.stop_event = threading.Event()
        self.cfgcpf, self.cfgcpt, self.fu, self.tu = fetch_counts_factors(SENSOR_IP)
        self.last_rdt_seq = None
        self.drop_est = 0
        self.last = None
        self.ts = []
        self.ux = []
        self.uy = []
        self.uz = []
        self.thy = []
        self.thz = []
        self.uyz = []
        plt.ion()
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
        self.fig.suptitle("Net F/T -> Needle Deflection (PDF method, live)")
        (self.l_ux,) = self.ax1.plot([], [], label="Ux (mm)")
        (self.l_uy,) = self.ax1.plot([], [], label="Uy (mm)")
        (self.l_uz,) = self.ax1.plot([], [], label="Uz (mm)")
        (self.l_uyz,) = self.ax1.plot([], [], label="Uyz (mm)")
        self.ax1.set_ylabel("Deflection (mm)")
        self.ax1.grid(True)
        self.ax1.legend(loc="upper left")
        (self.l_thy,) = self.ax2.plot([], [], label="θy (rad)")
        (self.l_thz,) = self.ax2.plot([], [], label="θz (rad)")
        self.ax2.set_xlabel("Time (s)")
        self.ax2.set_ylabel("Angle (rad)")
        self.ax2.grid(True)
        self.ax2.legend(loc="upper left")

    def prompt_user_inputs(self):
        print("请输入参数（回车采用默认值）：")
        try:
            f_fric = input("摩擦力 f_fric [N]（论文式 F=Fz-f）：")
            f_fric = float(f_fric) if f_fric.strip() != "" else 0.0
        except Exception:
            f_fric = 0.0
        try:
            f_ins = input(f"插入深度 f_insert [mm]（用于分布载荷，0~{self.model.L*1000:.1f}）：")
            f_ins = float(f_ins) if f_ins.strip() != "" else (self.model.L * 1000.0 / 2.0)
        except Exception:
            f_ins = self.model.L * 1000.0 / 2.0
        f_ins_m = max(0.0, min(f_ins / 1000.0, self.model.L))
        self.model.update_user_inputs(f_fric, f_ins_m)
        print(f"已设置：f_fric={self.model.f_fric:.3f} N, f_insert={self.model.f_insert*1000:.1f} mm, beta=30°, c=4.8 mm")

    def receiver_thread(self, buffered=False, sample_count=0):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("0.0.0.0", 0))
        sock.settimeout(1.0)
        cmd = CMD_START_BUFFERED if buffered else CMD_START_REALTIME
        sock.sendto(pack_rdt_request(cmd, sample_count), (SENSOR_IP, RDT_PORT))
        try:
            while not self.stop_event.is_set():
                try:
                    data, _ = sock.recvfrom(4096)
                except socket.timeout:
                    continue
                if len(data) % 36 != 0:
                    continue
                now = time.time()
                for i in range(0, len(data), 36):
                    rec = data[i:i+36]
                    rdt_seq, ft_seq, status, counts = unpack_rdt_record(rec)
                    if self.last_rdt_seq is not None:
                        expected = (self.last_rdt_seq + 1) & 0xFFFFFFFF
                        if rdt_seq != expected:
                            self.drop_est += (rdt_seq - expected) & 0xFFFFFFFF
                    self.last_rdt_seq = rdt_seq
                    fx_c, fy_c, fz_c, tx_c, ty_c, tz_c = counts
                    Fx_s = fx_c / self.cfgcpf
                    Fy_s = fy_c / self.cfgcpf
                    Fz_s = fz_c / self.cfgcpf
                    Mx_s = tx_c / self.cfgcpt
                    My_s = ty_c / self.cfgcpt
                    Mz_s = tz_c / self.cfgcpt
                    item = (now, rdt_seq, ft_seq, status, Fx_s, Fy_s, Fz_s, Mx_s, My_s, Mz_s)
                    try:
                        self.q.put_nowait(item)
                    except queue.Full:
                        try:
                            _ = self.q.get_nowait()
                            self.q.put_nowait(item)
                        except Exception:
                            pass
        finally:
            try:
                sock.sendto(pack_rdt_request(CMD_STOP, 0), (SENSOR_IP, RDT_PORT))
            except Exception:
                pass
            sock.close()

    def update_console(self, t_rel):
        if not self.last:
            return
        s = self.last
        line = (
            f"t={t_rel:7.2f}s | "
            f"SENSOR F=({s['Fx_s']:+7.3f},{s['Fy_s']:+7.3f},{s['Fz_s']:+7.3f})N "
            f"M=({s['Mx_s']:+7.4f},{s['My_s']:+7.4f},{s['Mz_s']:+7.4f})Nm || "
            f"MODEL Fm={s['Fm']:+6.3f}N Mm={s['Mm']:+6.3f}Nm Fnt={s['Fnt']:+6.3f}N | "
            f"Ux={s['Ux_mm']:+7.3f}mm Uy={s['Uy_mm']:+7.3f}mm Uz={s['Uz_mm']:+7.3f}mm Uyz={s['Uyz_mm']:+7.3f}mm | "
            f"θy={s['thy']:+.3e} θz={s['thz']:+.3e} | "
            f"status=0x{s['status']:08x} drop≈{self.drop_est}"
        )
        sys.stdout.write("\r" + line[:220].ljust(220))
        sys.stdout.flush()

    def update_plot(self):
        while self.ts and (self.ts[-1] - self.ts[0] > self.history_s):
            self.ts.pop(0)
            self.ux.pop(0); self.uy.pop(0); self.uz.pop(0); self.uyz.pop(0)
            self.thy.pop(0); self.thz.pop(0)
        self.l_ux.set_data(self.ts, self.ux)
        self.l_uy.set_data(self.ts, self.uy)
        self.l_uz.set_data(self.ts, self.uz)
        self.l_uyz.set_data(self.ts, self.uyz)
        self.ax1.relim()
        self.ax1.autoscale_view()
        self.l_thy.set_data(self.ts, self.thy)
        self.l_thz.set_data(self.ts, self.thz)
        self.ax2.relim()
        self.ax2.autoscale_view()
        plt.pause(0.001)

    def run(self, buffered=False, sample_count=0):
        print(f"Active config units: Force={self.fu}, Torque={self.tu} (assumed N/Nm)")
        print(f"Counts: cfgcpf={self.cfgcpf}, cfgcpt={self.cfgcpt}")
        print("Axis mapping: Model(x,y,z) = Sensor(z,x,y)")
        self.prompt_user_inputs()
        print("Starting RDT stream... Press Ctrl+C to stop.")
        t0 = time.time()
        last_ui = 0.0
        th = threading.Thread(
            target=self.receiver_thread,
            kwargs=dict(buffered=buffered, sample_count=sample_count),
            daemon=True
        )
        th.start()
        try:
            while True:
                now = time.time()
                t_rel = now - t0
                latest = None
                while True:
                    try:
                        latest = self.q.get_nowait()
                    except queue.Empty:
                        break
                if latest is not None:
                    (_, rdt_seq, ft_seq, status, Fx_s, Fy_s, Fz_s, Mx_s, My_s, Mz_s) = latest
                    Fx_m, Fy_m, Fz_m, Mx_m, My_m, Mz_m = sensor_to_model_axes(
                        Fx_s, Fy_s, Fz_s, Mx_s, My_s, Mz_s
                    )
                    out = self.model.compute_deflection(Fx_m, Fy_m, Fz_m, Mx_m, My_m, Mz_m)
                    self.last = {
                        "Fx_s": Fx_s, "Fy_s": Fy_s, "Fz_s": Fz_s,
                        "Mx_s": Mx_s, "My_s": My_s, "Mz_s": Mz_s,
                        "Fm": out["Fm"], "Mm": out["Mm"], "Fnt": out["Fnt"],
                        "Ux_mm": out["Ux"] * 1000.0,
                        "Uy_mm": out["Uy"] * 1000.0,
                        "Uz_mm": out["Uz"] * 1000.0,
                        "Uyz_mm": out["Uyz"] * 1000.0,
                        "thy": out["theta_y"],
                        "thz": out["theta_z"],
                        "status": status,
                        "rdt_seq": rdt_seq,
                        "ft_seq": ft_seq,
                    }
                    self.ts.append(t_rel)
                    self.ux.append(out["Ux"] * 1000.0)
                    self.uy.append(out["Uy"] * 1000.0)
                    self.uz.append(out["Uz"] * 1000.0)
                    self.uyz.append(out["Uyz"] * 1000.0)
                    self.thy.append(out["theta_y"])
                    self.thz.append(out["theta_z"])
                if t_rel - last_ui >= self.display_period_s:
                    last_ui = t_rel
                    self.update_console(t_rel)
                    self.update_plot()
                time.sleep(0.001)
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            self.stop_event.set()
            th.join(timeout=2.0)
            print("Stopped.")

if __name__ == "__main__":
    processor = LiveProcessor(needle_length_m=0.200, display_period_s=0.1, history_s=10.0)
    processor.run(buffered=False, sample_count=0)