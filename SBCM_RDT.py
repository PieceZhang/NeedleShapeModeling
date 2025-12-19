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

class NeedleDeflectionModel:
    def __init__(self):
        self.needle_length = 200e-3
        self.needle_diameter = 1.27e-3
        self.E = 200e9
        self.G = 80e9
        self.mu = 0.28
        self.A = np.pi * (self.needle_diameter / 2) ** 2
        self.I = np.pi * self.needle_diameter ** 4 / 64
        self.J = 2 * self.I

        self.H1 = np.array([
            [12.0, -6.0, 0.0, 0.0],
            [-6.0, 4.0, 0.0, 0.0],
            [0.0, 0.0, 12.0, 6.0],
            [0.0, 0.0, 6.0, 4.0]
        ], dtype=float)

        self.H2 = np.array([
            [-3.0/5, 1.0/20, 0.0, 0.0],
            [1.0/20, -1.0/15, 0.0, 0.0],
            [0.0, 0.0, -3.0/5, -1.0/20],
            [0.0, 0.0, -1.0/20, -1.0/15]
        ], dtype=float)

        self.H3 = np.array([
            [0.0, 0.0, 0.0, -1.0/2],
            [0.0, 0.0, -1.0/2, -1.0/4],
            [0.0, -1.0/2, 0., 0.0],
            [-1.0/2, -1.0/4, 0.0, 0.0]
        ], dtype=float)

        self.H4 = np.array([
            [1.0/700, -1.0/1400, 0.0, 0.0],
            [-1.0/1400, 11.0/6300, 0.0, 0.0],
            [0.0, 0.0, 1.0/700, 1.0/1400],
            [0.0, 0.0, 1.0/1400, 11.0/6300]
        ], dtype=float)

        self.H5 = np.array([
            [0.0, 0.0, 0.0, 1.0/60],
            [0.0, 0.0, 1.0/60, 0.0],
            [0.0, 1.0/60, 0.0, 0.0],
            [1.0/60, 0.0, 0.0, 0.0]
        ], dtype=float)

        self.H6 = np.array([
            [1.0/5, -1.0/10, 0.0, 0.0],
            [-1.0/10, 1.0/20, 0.0, 0.0],
            [0.0, 0.0, 1.0/5, 1.0/10],
            [0.0, 0.0, 1.0/10, 1.0/20]
        ], dtype=float)

        self.H7 = np.array([
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0]
        ], dtype=float)

    def normalize_parameters(self, F_x, F_y, F_z, M_x, M_y, M_z, L, n=1):
        f_x = F_x * L ** 2 / (n ** 2 * self.E * self.I)
        f_y = F_y * L ** 2 / (n ** 2 * self.E * self.I)
        f_z = F_z * L ** 2 / (n ** 2 * self.E * self.I)
        m_x = M_x * L / (n * self.E * self.I)
        m_y = M_y * L / (n * self.E * self.I)
        m_z = M_z * L / (n * self.E * self.I)
        return f_x, f_y, f_z, m_x, m_y, m_z

    def denormalize_displacement(self, u_y, theta_z, u_z, theta_y, L, n=1):
        U_y = u_y * L / n
        U_z = u_z * L / n
        Theta_y = theta_y
        Theta_z = theta_z
        return U_y, Theta_z, U_z, Theta_y

    def calculate_residual(self, v, f_x, f_y, f_z, m_x, m_y, m_z):
        u_y, theta_z, u_z, theta_y = v
        m_xd = m_x + theta_z * m_y + theta_y * m_z
        g = np.array([f_y, m_z, f_z, m_y])
        term1 = self.H1 @ v
        term2 = (2 * f_x * self.H2 + m_xd * (2 * self.H3 + self.H7)) @ v
        term3 = (f_x ** 2 * self.H4 + m_xd * f_z * self.H5 + m_xd ** 2 * self.H6) @ v
        return g - (term1 - term2 - term3)

    def calculate_jacobian(self, v, f_x, f_y, f_z, m_x, m_y, m_z):
        u_y, theta_z, u_z, theta_y = v
        m_xd = m_x + theta_z * m_y + theta_y * m_z
        J = -self.H1.copy()
        J += (2 * f_x * self.H2 + m_xd * (2 * self.H3 + self.H7))
        J += (f_x ** 2 * self.H4 + m_xd * f_z * self.H5 + m_xd ** 2 * self.H6)
        dm_xd_dv = np.array([0, m_y, 0, m_z])
        term_mxd_linear = (2 * self.H3 + self.H7) @ v
        term_mxd_quadratic = (f_z * self.H5 + 2 * m_xd * self.H6) @ v
        J += np.outer(term_mxd_linear, dm_xd_dv)
        J += np.outer(term_mxd_quadratic, dm_xd_dv)
        return J

    def calculate_tip_deflection_newton(self, F_x, F_y, F_z, M_x, M_y, M_z, L):
        f_x, f_y, f_z, m_x, m_y, m_z = self.normalize_parameters(F_x, F_y, F_z, M_x, M_y, M_z, L)
        try:
            v_linear = np.linalg.solve(self.H1, np.array([f_y, m_z, f_z, m_y]))
            v = v_linear * 0.5
        except Exception:
            v = np.array([0.001, 0.001, 0.001, 0.001])

        max_iterations = 20
        tolerance = 1e-8
        for _ in range(max_iterations):
            F_v = self.calculate_residual(v, f_x, f_y, f_z, m_x, m_y, m_z)
            J = self.calculate_jacobian(v, f_x, f_y, f_z, m_x, m_y, m_z)
            if np.linalg.norm(F_v) < tolerance:
                break

            try:
                delta_v = np.linalg.solve(J, -F_v)
            except np.linalg.LinAlgError:
                delta_v = np.linalg.lstsq(J, -F_v, rcond=None)[0]

            v_old = v.copy()
            v = v + delta_v

            if np.linalg.norm(delta_v) > 10.0:
                v = v_old + 0.5 * delta_v

            if np.any(np.isnan(v)) or np.any(np.isinf(v)):
                v = np.array([0.0001, 0.0001, 0.0001, 0.0001])

        u_y, theta_z, u_z, theta_y = v
        U_y, Theta_z, U_z, Theta_y = self.denormalize_displacement(u_y, theta_z, u_z, theta_y, L)
        U_x = (F_x * L) / (self.E * self.A)
        return U_x, U_y, U_z, Theta_y, Theta_z


# ===== Net F/T RDT per 9620-05-NET FT.pdf section 9 =====
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
    """
    你给的对应关系：
      Sensor Fx -> Model Fy
      Sensor Fy -> Model Fz
      Sensor Fz -> Model Fx
    => Model (x,y,z) = Sensor (z,x,y)
    力矩同理。
    """
    Fx_m = Fz_s
    Fy_m = Fx_s
    Fz_m = Fy_s
    Mx_m = Mz_s
    My_m = Mx_s
    Mz_m = My_s
    return Fx_m, Fy_m, Fz_m, Mx_m, My_m, Mz_m


class LiveProcessor:
    def __init__(self, needle_length_m=0.200, display_period_s=0.1, history_s=10.0):
        self.model = NeedleDeflectionModel()
        self.model.needle_length = needle_length_m

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

        plt.ion()
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
        self.fig.suptitle("Net F/T (9105-NETBA) RDT -> Needle Deflection (Live)")

        (self.l_ux,) = self.ax1.plot([], [], label="Ux (mm)")
        (self.l_uy,) = self.ax1.plot([], [], label="Uy (mm)")
        (self.l_uz,) = self.ax1.plot([], [], label="Uz (mm)")
        self.ax1.set_ylabel("Deflection (mm)")
        self.ax1.grid(True)
        self.ax1.legend(loc="upper left")

        (self.l_thy,) = self.ax2.plot([], [], label="θy (rad)")
        (self.l_thz,) = self.ax2.plot([], [], label="θz (rad)")
        self.ax2.set_xlabel("Time (s)")
        self.ax2.set_ylabel("Angle (rad)")
        self.ax2.grid(True)
        self.ax2.legend(loc="upper left")

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

                    # scale to N / Nm (你已确认单位就是 N/Nm)
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
                        # drop one oldest then try again
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
            f"MODEL F=({s['Fx_m']:+7.3f},{s['Fy_m']:+7.3f},{s['Fz_m']:+7.3f})N "
            f"M=({s['Mx_m']:+7.4f},{s['My_m']:+7.4f},{s['Mz_m']:+7.4f})Nm | "
            f"Ux={s['Ux_mm']:+7.3f}mm Uy={s['Uy_mm']:+7.3f}mm Uz={s['Uz_mm']:+7.3f}mm | "
            f"θy={s['thy']:+.3e} θz={s['thz']:+.3e} | "
            f"status=0x{s['status']:08x} drop≈{self.drop_est}"
        )
        sys.stdout.write("\r" + line[:220].ljust(220))
        sys.stdout.flush()

    def update_plot(self):
        while self.ts and (self.ts[-1] - self.ts[0] > self.history_s):
            self.ts.pop(0)
            self.ux.pop(0); self.uy.pop(0); self.uz.pop(0)
            self.thy.pop(0); self.thz.pop(0)

        self.l_ux.set_data(self.ts, self.ux)
        self.l_uy.set_data(self.ts, self.uy)
        self.l_uz.set_data(self.ts, self.uz)
        self.ax1.relim(); self.ax1.autoscale_view()

        self.l_thy.set_data(self.ts, self.thy)
        self.l_thz.set_data(self.ts, self.thz)
        self.ax2.relim(); self.ax2.autoscale_view()

        plt.pause(0.001)

    def run(self, buffered=False, sample_count=0):
        print(f"Active config units: Force={self.fu}, Torque={self.tu} (you said N/Nm)")
        print(f"Counts: cfgcpf={self.cfgcpf}, cfgcpt={self.cfgcpt}")
        print("Axis mapping: Model(x,y,z) = Sensor(z,x,y)")
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

                    # remap axes for your model
                    Fx_m, Fy_m, Fz_m, Mx_m, My_m, Mz_m = sensor_to_model_axes(
                        Fx_s, Fy_s, Fz_s, Mx_s, My_s, Mz_s
                    )

                    # compute deflection
                    Ux, Uy, Uz, Thy, Thz = self.model.calculate_tip_deflection_newton(
                        Fx_m, Fy_m, Fz_m, Mx_m, My_m, Mz_m, self.model.needle_length
                    )

                    self.last = {
                        "Fx_s": Fx_s, "Fy_s": Fy_s, "Fz_s": Fz_s,
                        "Mx_s": Mx_s, "My_s": My_s, "Mz_s": Mz_s,
                        "Fx_m": Fx_m, "Fy_m": Fy_m, "Fz_m": Fz_m,
                        "Mx_m": Mx_m, "My_m": My_m, "Mz_m": Mz_m,
                        "Ux_mm": Ux * 1000.0,
                        "Uy_mm": Uy * 1000.0,
                        "Uz_mm": Uz * 1000.0,
                        "thy": Thy,
                        "thz": Thz,
                        "status": status,
                        "rdt_seq": rdt_seq,
                        "ft_seq": ft_seq,
                    }

                    self.ts.append(t_rel)
                    self.ux.append(Ux * 1000.0)
                    self.uy.append(Uy * 1000.0)
                    self.uz.append(Uz * 1000.0)
                    self.thy.append(Thy)
                    self.thz.append(Thz)

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