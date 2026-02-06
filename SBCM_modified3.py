import socket
import struct
import time
import threading
import queue
import sys
import urllib.request
import xml.etree.ElementTree as ET
from collections import deque

import numpy as np
import matplotlib.pyplot as plt


class NeedleDeflectionModel:
    def __init__(self, calibration_factor=1.0):
        self.needle_length = 200e-3
        self.needle_diameter = 1.27e-3
        self.E = 200e9
        self.G = 80e9
        self.mu = 0.28
        self.A = np.pi * (self.needle_diameter / 2) ** 2
        self.I = np.pi * self.needle_diameter ** 4 / 64
        self.J = 2 * self.I
        self.calibration_factor = calibration_factor

        self.H1 = np.array([
            [12.0, -6.0, 0.0, 0.0],
            [-6.0, 4.0, 0.0, 0.0],
            [0.0, 0.0, 12.0, 6.0],
            [0.0, 0.0, 6.0, 4.0]
        ], dtype=float)

        self.H2 = np.array([
            [-3.0 / 5, 1.0 / 20, 0.0, 0.0],
            [1.0 / 20, -1.0 / 15, 0.0, 0.0],
            [0.0, 0.0, -3.0 / 5, -1.0 / 20],
            [0.0, 0.0, -1.0 / 20, -1.0 / 15]
        ], dtype=float)

        self.H3 = np.array([
            [0.0, 0.0, 0.0, -1.0 / 2],
            [0.0, 0.0, -1.0 / 2, -1.0 / 4],
            [0.0, -1.0 / 2, 0., 0.0],
            [-1.0 / 2, -1.0 / 4, 0.0, 0.0]
        ], dtype=float)

        self.H4 = np.array([
            [1.0 / 700, -1.0 / 1400, 0.0, 0.0],
            [-1.0 / 1400, 11.0 / 6300, 0.0, 0.0],
            [0.0, 0.0, 1.0 / 700, 1.0 / 1400],
            [0.0, 0.0, 1.0 / 1400, 11.0 / 6300]
        ], dtype=float)

        self.H5 = np.array([
            [0.0, 0.0, 0.0, 1.0 / 60],
            [0.0, 0.0, 1.0 / 60, 0.0],
            [0.0, 1.0 / 60, 0.0, 0.0],
            [1.0 / 60, 0.0, 0.0, 0.0]
        ], dtype=float)

        self.H6 = np.array([
            [1.0 / 5, -1.0 / 10, 0.0, 0.0],
            [-1.0 / 10, 1.0 / 20, 0.0, 0.0],
            [0.0, 0.0, 1.0 / 5, 1.0 / 10],
            [0.0, 0.0, 1.0 / 10, 1.0 / 20]
        ], dtype=float)

        self.H7 = np.array([
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0]
        ], dtype=float)

    def set_calibration_factor(self, factor):
        self.calibration_factor = factor

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
        try:
            if (abs(F_x) > 100 or abs(F_y) > 100 or abs(F_z) > 100 or
                    abs(M_x) > 10 or abs(M_y) > 10 or abs(M_z) > 10):
                return 0, 0, 0, 0, 0

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

            U_x = U_x * self.calibration_factor
            U_y = U_y * self.calibration_factor
            U_z = U_z * self.calibration_factor
            Theta_y = Theta_y * self.calibration_factor
            Theta_z = Theta_z * self.calibration_factor

            return U_x, U_y, U_z, Theta_y, Theta_z

        except Exception as e:
            return 0, 0, 0, 0, 0


SENSOR_IP = "192.168.1.1"
RDT_PORT = 49152

HDR = 0x1234
CMD_STOP = 0x0000
CMD_START_REALTIME = 0x0002
CMD_START_BUFFERED = 0x0003


def fetch_counts_factors(sensor_ip: str):
    try:
        url = f"http://{sensor_ip}/netftapi2.xml"
        with urllib.request.urlopen(url, timeout=2.0) as resp:
            xml_bytes = resp.read()
        root = ET.fromstring(xml_bytes)
        cfgcpf = int(root.findtext("cfgcpf"))
        cfgcpt = int(root.findtext("cfgcpt"))
        fu = root.findtext("scfgfu")
        tu = root.findtext("scfgtu")
        return cfgcpf, cfgcpt, fu, tu
    except Exception as e:
        print(f"获取传感器配置失败: {e}")
        return 100000, 100000, "N", "Nm"


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


class LiveProcessor:
    def __init__(self, needle_length_m=0.200, display_period_s=0.1, history_s=10.0, calibration_factor=1.0):
        self.calibration_factor = calibration_factor
        self.model = NeedleDeflectionModel(calibration_factor)
        self.model.needle_length = needle_length_m

        self.display_period_s = display_period_s
        self.history_s = history_s

        self.q = queue.Queue(maxsize=3000)
        self.stop_event = threading.Event()

        try:
            self.cfgcpf, self.cfgcpt, self.fu, self.tu = fetch_counts_factors(SENSOR_IP)
        except Exception as e:
            print(f"警告: 使用默认传感器配置: {e}")
            self.cfgcpf, self.cfgcpt, self.fu, self.tu = 100000, 100000, "N", "Nm"

        self.last_rdt_seq = None
        self.drop_est = 0
        self.error_count = 0
        self.success_count = 0

        self.last = None

        self.ts = []
        self.ux = []
        self.uy = []
        self.uz = []
        self.thy = []
        self.thz = []
        self.utotal = []
        self.utotal_filtered = []

        self.fx_history = []
        self.fy_history = []
        self.fz_history = []
        self.mx_history = []
        self.my_history = []
        self.mz_history = []

        self.filter_window = 5
        self.force_buffer = deque(maxlen=self.filter_window)
        self.moment_buffer = deque(maxlen=self.filter_window)

        self.last_valid_data = None

        plt.ion()
        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.suptitle(f"Net F/T RDT -> Needle Deflection (Live) [Calibration Factor: {self.calibration_factor}]")

        (self.l_ux,) = self.ax1.plot([], [], label="Ux (mm)", color='blue', linewidth=1.5)
        (self.l_uy,) = self.ax1.plot([], [], label="Uy (mm)", color='green', linewidth=1.5)
        (self.l_uz,) = self.ax1.plot([], [], label="Uz (mm)", color='red', linewidth=1.5)
        (self.l_utotal,) = self.ax1.plot([], [], label="U_total (mm)", color='purple', linewidth=2, linestyle='--')
        self.ax1.set_ylabel("Displacement (mm)")
        self.ax1.set_title("Displacement Components")
        self.ax1.grid(True, alpha=0.3)
        self.ax1.legend(loc="upper left", fontsize=9)

        (self.l_thy,) = self.ax2.plot([], [], label="θy (rad)", color='orange', linewidth=1.5)
        (self.l_thz,) = self.ax2.plot([], [], label="θz (rad)", color='brown', linewidth=1.5)
        self.ax2.set_ylabel("Angle (rad)")
        self.ax2.set_title("Angular Displacement")
        self.ax2.grid(True, alpha=0.3)
        self.ax2.legend(loc="upper left", fontsize=9)

        (self.l_utotal_alone,) = self.ax3.plot([], [], label=f"U_total (CF={self.calibration_factor})",
                                               color='darkviolet', linewidth=2.5)
        (self.l_utotal_filtered_line,) = self.ax3.plot([], [], label=f"Filtered (window={self.filter_window})",
                                                       color='cyan', linewidth=1.5, alpha=0.7)
        self.ax3.set_xlabel("Time (s)")
        self.ax3.set_ylabel("Total Displacement (mm)")
        self.ax3.set_title("Total Needle Deflection")
        self.ax3.grid(True, alpha=0.3)
        self.ax3.legend(loc="upper left", fontsize=10)

        self.force_lines = []
        colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown']
        labels = ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']
        for i in range(6):
            line, = self.ax4.plot([], [], label=labels[i], color=colors[i], linewidth=1.2)
            self.force_lines.append(line)
        self.ax4.set_xlabel("Time (s)")
        self.ax4.set_ylabel("Force (N) / Torque (Nm)")
        self.ax4.set_title("Raw Sensor Data")
        self.ax4.grid(True, alpha=0.3)
        self.ax4.legend(loc="upper left", fontsize=8, ncol=2)

        plt.tight_layout()

    def set_calibration_factor(self, factor):
        self.calibration_factor = factor
        self.model.set_calibration_factor(factor)
        self.fig.suptitle(f"Net F/T RDT -> Needle Deflection (Live) [Calibration Factor: {self.calibration_factor}]")

    def apply_filter(self, force_data):
        self.force_buffer.append(force_data[:3])
        self.moment_buffer.append(force_data[3:])

        if len(self.force_buffer) == 0:
            return force_data

        filtered_force = np.mean(self.force_buffer, axis=0)
        filtered_moment = np.mean(self.moment_buffer, axis=0)

        return np.concatenate([filtered_force, filtered_moment])

    def receiver_thread(self, buffered=False, sample_count=0):
        sock = None
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.bind(("0.0.0.0", 0))
            sock.settimeout(1.0)

            cmd = CMD_START_BUFFERED if buffered else CMD_START_REALTIME
            sock.sendto(pack_rdt_request(cmd, sample_count), (SENSOR_IP, RDT_PORT))

            while not self.stop_event.is_set():
                try:
                    data, _ = sock.recvfrom(4096)
                except socket.timeout:
                    continue
                except Exception as e:
                    print(f"接收数据错误: {e}")
                    break

                if len(data) % 36 != 0:
                    continue

                now = time.time()
                for i in range(0, len(data), 36):
                    rec = data[i:i + 36]
                    try:
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
                    except Exception as e:
                        print(f"解析数据包错误: {e}")
        except Exception as e:
            print(f"接收线程错误: {e}")
        finally:
            if sock:
                try:
                    sock.sendto(pack_rdt_request(CMD_STOP, 0), (SENSOR_IP, RDT_PORT))
                except Exception:
                    pass
                sock.close()

    def update_console(self, t_rel):
        if not self.last:
            return
        s = self.last
        error_rate = self.error_count / max(self.success_count + self.error_count, 1) * 100
        line = (
            f"t={t_rel:7.2f}s | "
            f"CF={self.calibration_factor:.3f} | "
            f"U_total={s['U_total']:7.3f}mm | "
            f"Ux={s['Ux_mm']:+7.3f}mm Uy={s['Uy_mm']:+7.3f}mm Uz={s['Uz_mm']:+7.3f}mm | "
            f"F=({s['Fx_s']:+6.2f},{s['Fy_s']:+6.2f},{s['Fz_s']:+6.2f})N | "
            f"Errors: {self.error_count}({error_rate:.1f}%) | "
            f"Drop≈{self.drop_est}"
        )
        sys.stdout.write("\r" + line[:220].ljust(220))
        sys.stdout.flush()

    def update_plot(self):
        while self.ts and (self.ts[-1] - self.ts[0] > self.history_s):
            self.ts.pop(0)
            self.ux.pop(0);
            self.uy.pop(0);
            self.uz.pop(0)
            self.thy.pop(0);
            self.thz.pop(0)
            self.utotal.pop(0)
            if self.utotal_filtered:
                self.utotal_filtered.pop(0)

            if self.fx_history:
                self.fx_history.pop(0);
                self.fy_history.pop(0);
                self.fz_history.pop(0)
                self.mx_history.pop(0);
                self.my_history.pop(0);
                self.mz_history.pop(0)

        self.l_ux.set_data(self.ts, self.ux)
        self.l_uy.set_data(self.ts, self.uy)
        self.l_uz.set_data(self.ts, self.uz)
        self.l_utotal.set_data(self.ts, self.utotal)
        self.ax1.relim()
        self.ax1.autoscale_view()

        self.l_thy.set_data(self.ts, self.thy)
        self.l_thz.set_data(self.ts, self.thz)
        self.ax2.relim()
        self.ax2.autoscale_view()

        self.l_utotal_alone.set_data(self.ts, self.utotal)
        if self.utotal_filtered and len(self.utotal_filtered) == len(self.ts):
            self.l_utotal_filtered_line.set_data(self.ts, self.utotal_filtered)
        self.ax3.relim()
        self.ax3.autoscale_view()

        if len(self.ts) == len(self.fx_history):
            sensor_data = [self.fx_history, self.fy_history, self.fz_history,
                           self.mx_history, self.my_history, self.mz_history]
            for i, line in enumerate(self.force_lines):
                line.set_data(self.ts, sensor_data[i])
            self.ax4.relim()
            self.ax4.autoscale_view()

        if self.ts and self.utotal:
            current_utotal = self.utotal[-1]
            if self.utotal_filtered:
                current_filtered = self.utotal_filtered[-1]
                self.ax3.set_title(
                    f"Total Needle Deflection [CF={self.calibration_factor}]\nRaw: {current_utotal:.3f}mm, Filtered: {current_filtered:.3f}mm")
            else:
                self.ax3.set_title(
                    f"Total Needle Deflection [CF={self.calibration_factor}]\nCurrent: {current_utotal:.3f} mm")
            current_values = f"CF={self.calibration_factor} | Ux:{self.ux[-1]:.2f}, Uy:{self.uy[-1]:.2f}, Uz:{self.uz[-1]:.2f}, U:{self.utotal[-1]:.2f} mm"
            self.ax1.set_title(f"Displacement Components\n{current_values}")

        try:
            plt.pause(0.001)
        except Exception:
            pass

    def run(self, buffered=False, sample_count=0):
        print(f"标定系数 (Calibration Factor): {self.calibration_factor}")
        print(f"传感器单位: 力={self.fu}, 力矩={self.tu}")
        print(f"转换系数: cfgcpf={self.cfgcpf}, cfgcpt={self.cfgcpt}")
        print("坐标映射: Model(x,y,z) = Sensor(z,x,y)")
        print(f"滤波窗口: {self.filter_window}个样本")
        print("启动RDT数据流... 按Ctrl+C停止。")
        print("-" * 80)

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

                collected_data = []
                while not self.q.empty():
                    try:
                        latest = self.q.get_nowait()
                        collected_data.append(latest)
                    except queue.Empty:
                        break

                if collected_data:
                    latest = collected_data[-1]
                    try:
                        (_, rdt_seq, ft_seq, status, Fx_s, Fy_s, Fz_s, Mx_s, My_s, Mz_s) = latest

                        self.fx_history.append(Fx_s)
                        self.fy_history.append(Fy_s)
                        self.fz_history.append(Fz_s)
                        self.mx_history.append(Mx_s)
                        self.my_history.append(My_s)
                        self.mz_history.append(Mz_s)

                        force_data = np.array([Fx_s, Fy_s, Fz_s, Mx_s, My_s, Mz_s])
                        filtered_data = self.apply_filter(force_data)

                        Fx_s_filt, Fy_s_filt, Fz_s_filt, Mx_s_filt, My_s_filt, Mz_s_filt = filtered_data

                        Fx_m, Fy_m, Fz_m, Mx_m, My_m, Mz_m = sensor_to_model_axes(
                            Fx_s_filt, Fy_s_filt, Fz_s_filt, Mx_s_filt, My_s_filt, Mz_s_filt
                        )

                        Ux, Uy, Uz, Thy, Thz = self.model.calculate_tip_deflection_newton(
                            Fx_m, Fy_m, Fz_m, Mx_m, My_m, Mz_m, self.model.needle_length
                        )

                        if Ux == 0 and Uy == 0 and Uz == 0 and Thy == 0 and Thz == 0:
                            self.error_count += 1
                            if self.last_valid_data:
                                Ux, Uy, Uz, Thy, Thz, U_total = self.last_valid_data
                            else:
                                continue
                        else:
                            Uy_mm = Uy * 1000
                            Uz_mm = Uz * 1000
                            U_total = np.sqrt(Uy_mm ** 2 + Uz_mm ** 2)
                            self.success_count += 1
                            self.last_valid_data = (Ux * 1000, Uy_mm, Uz_mm, Thy, Thz, U_total)

                        self.last = {
                            "Fx_s": Fx_s, "Fy_s": Fy_s, "Fz_s": Fz_s,
                            "Mx_s": Mx_s, "My_s": My_s, "Mz_s": Mz_s,
                            "Fx_m": Fx_m, "Fy_m": Fy_m, "Fz_m": Fz_m,
                            "Mx_m": Mx_m, "My_m": My_m, "Mz_m": Mz_m,
                            "Ux_mm": Ux * 1000,
                            "Uy_mm": Uy_mm,
                            "Uz_mm": Uz_mm,
                            "U_total": U_total,
                            "thy": Thy,
                            "thz": Thz,
                            "status": status,
                            "rdt_seq": rdt_seq,
                            "ft_seq": ft_seq,
                        }

                        self.ts.append(t_rel)
                        self.ux.append(Ux * 1000)
                        self.uy.append(Uy_mm)
                        self.uz.append(Uz_mm)
                        self.thy.append(Thy)
                        self.thz.append(Thz)
                        self.utotal.append(U_total)

                        if len(self.utotal) >= self.filter_window:
                            filtered_utotal = np.mean(self.utotal[-self.filter_window:])
                            self.utotal_filtered.append(filtered_utotal)

                    except Exception as e:
                        self.error_count += 1

                if t_rel - last_ui >= self.display_period_s:
                    last_ui = t_rel
                    self.update_console(t_rel)
                    self.update_plot()

                time.sleep(0.001)

        except KeyboardInterrupt:
            print("\n\n正在停止...")
            self.stop_event.set()
            th.join(timeout=2.0)
            self.export_data()
            print("程序已停止。")
        except Exception as e:
            print(f"\n程序错误: {e}")
            self.stop_event.set()
            th.join(timeout=2.0)
            self.export_data()
        finally:
            if not self.stop_event.is_set():
                self.stop_event.set()
                th.join(timeout=2.0)

    def export_data(self, filename="needle_deflection_data.csv"):
        import csv

        min_len = min(len(self.ts), len(self.ux), len(self.uy), len(self.uz),
                      len(self.utotal), len(self.thy), len(self.thz))

        if min_len == 0:
            print("没有数据可导出")
            return

        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Calibration_Factor', self.calibration_factor])
            writer.writerow(['Time(s)', 'Ux(mm)', 'Uy(mm)', 'Uz(mm)', 'U_total(mm)',
                             'U_total_filtered(mm)', 'Theta_y(rad)', 'Theta_z(rad)',
                             'Fx_s(N)', 'Fy_s(N)', 'Fz_s(N)', 'Mx_s(Nm)', 'My_s(Nm)', 'Mz_s(Nm)'])

            for i in range(min_len):
                fx = self.fx_history[i] if i < len(self.fx_history) else 0
                fy = self.fy_history[i] if i < len(self.fy_history) else 0
                fz = self.fz_history[i] if i < len(self.fz_history) else 0
                mx = self.mx_history[i] if i < len(self.mx_history) else 0
                my = self.my_history[i] if i < len(self.my_history) else 0
                mz = self.mz_history[i] if i < len(self.mz_history) else 0

                utotal_filtered = self.utotal_filtered[i] if i < len(self.utotal_filtered) else self.utotal[i]

                writer.writerow([
                    self.ts[i], self.ux[i], self.uy[i], self.uz[i], self.utotal[i],
                    utotal_filtered, self.thy[i], self.thz[i], fx, fy, fz, mx, my, mz
                ])

        print(f"数据已导出到 {filename}，共 {min_len} 条记录")
        print(f"标定系数: {self.calibration_factor}")
        print(f"成功计算: {self.success_count}, 错误: {self.error_count}, 丢包: {self.drop_est}")


if __name__ == "__main__":
    calibration_factor = 0.69

    try:
        processor = LiveProcessor(
            needle_length_m=0.200,
            display_period_s=0.05,
            history_s=15.0,
            calibration_factor=calibration_factor
        )
        processor.run(buffered=False, sample_count=0)
    except Exception as e:
        print(f"程序启动失败: {e}")