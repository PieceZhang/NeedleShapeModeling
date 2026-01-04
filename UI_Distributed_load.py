import sys
import numpy as np
from PyQt5 import QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

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

def compute_geometry_pdf(L, result, segments=120):
    """三次 Hermite 插值生成挠曲曲线（基于 PDF 模型输出）"""
    safe_L = max(L, 1e-6)
    s = np.linspace(0, safe_L, segments)
    t = s / safe_L
    t2 = t * t
    t3 = t2 * t
    h00 = 2 * t3 - 3 * t2 + 1
    h10 = t3 - 2 * t2 + t
    h01 = -2 * t3 + 3 * t2
    h11 = t3 - t2
    Ux, Uy, Uz = result["Ux"], result["Uy"], result["Uz"]
    th_y, th_z = result["theta_y"], result["theta_z"]
    y = h10 * (0.0 * safe_L) + h01 * Uy + h11 * (th_z * safe_L)
    z = h10 * (0.0 * safe_L) + h01 * Uz + h11 * (-th_y * safe_L)
    x = s + Ux * t
    curve = np.column_stack((x, y, z)) * 1000  # m -> mm
    baseline = np.column_stack((s, np.zeros_like(s), np.zeros_like(s))) * 1000
    return curve, baseline

class ParamSlider(QtWidgets.QWidget):
    value_changed = QtCore.pyqtSignal(float)
    def __init__(self, title, min_val, max_val, step, initial, unit="", precision=2, parent=None):
        super().__init__(parent)
        self.min_val, self.max_val, self.step = min_val, max_val, step
        self.unit, self.precision = unit, precision
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel(f"{title} ({min_val}~{max_val} {unit})"))
        h = QtWidgets.QHBoxLayout()
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        ticks = int(round((max_val - min_val) / step))
        self.slider.setMinimum(0); self.slider.setMaximum(ticks)
        self.slider.setValue(int(round((initial - min_val) / step)))
        h.addWidget(self.slider)
        self.value_label = QtWidgets.QLabel(); self.value_label.setFixedWidth(90); h.addWidget(self.value_label)
        layout.addLayout(h)
        self.slider.valueChanged.connect(self._emit_value)
        self._emit_value(self.slider.value())
    def _emit_value(self, v):
        val = self.min_val + v * self.step
        fmt = f"{{:.{self.precision}f}}"
        txt = fmt.format(val) + (f" {self.unit}" if self.unit else "")
        self.value_label.setText(txt)
        self.value_changed.emit(val)
    def value(self):
        return self.min_val + self.slider.value() * self.step

class NeedleCanvas(FigureCanvas):
    def __init__(self, parent=None):
        fig = Figure(figsize=(5, 4), tight_layout=True)
        super().__init__(fig)
        self.ax = fig.add_subplot(111, projection="3d")
        fig.patch.set_facecolor("#0f172a")
        self.ax.set_facecolor("#0b1120")
        self.ax.set_xlabel("X (mm)"); self.ax.set_ylabel("Y (mm)"); self.ax.set_zlabel("Z (mm)")
        self.ax.view_init(elev=22, azim=-60)
        self._view = self._capture()
        for evt in ("button_release_event", "scroll_event", "key_release_event"):
            self.mpl_connect(evt, self._on_change)
    def _capture(self):
        return {"elev": getattr(self.ax, "elev", 30.0), "azim": getattr(self.ax, "azim", -60.0), "dist": getattr(self.ax, "dist", 10.0)}
    def _apply(self, st):
        self.ax.view_init(elev=st.get("elev", 30.0), azim=st.get("azim", -60.0))
        if st.get("dist") is not None:
            self.ax.dist = st["dist"]
    def _on_change(self, _):
        self._view = self._capture()
    def update_scene(self, curve, baseline, insert_start_m=None):
        st = self._view.copy()
        self.ax.cla()
        self.ax.set_xlabel("X (mm)"); self.ax.set_ylabel("Y (mm)"); self.ax.set_zlabel("Z (mm)")
        if curve is not None and len(curve) > 0:
            # 外露段（未插入）虚线
            self.ax.plot(baseline[:, 0], baseline[:, 1], baseline[:, 2], "--", color="#94a3b8", linewidth=1.5, label="Baseline")
            if insert_start_m is not None:
                mask_ins = (curve[:, 0] / 1000.0) >= insert_start_m
            else:
                mask_ins = np.ones(len(curve), dtype=bool)
            mask_out = ~mask_ins
            if mask_out.any():
                self.ax.plot(curve[mask_out, 0], curve[mask_out, 1], curve[mask_out, 2],
                             color="#cbd5e1", linewidth=2, label="外露段")
            if mask_ins.any():
                self.ax.plot(curve[mask_ins, 0], curve[mask_ins, 1], curve[mask_ins, 2],
                             color="#38bdf8", linewidth=3, label="插入段")
            self.ax.scatter(curve[-1, 0], curve[-1, 1], curve[-1, 2], color="#f97316", s=40, label="Tip")
            all_pts = np.vstack((curve, baseline))
            max_range = (all_pts.max(axis=0) - all_pts.min(axis=0)).max()
            center = all_pts.mean(axis=0); half = max_range / 2 if max_range > 0 else 1
            self.ax.set_xlim(center[0] - half, center[0] + half)
            self.ax.set_ylim(center[1] - half, center[1] + half)
            self.ax.set_zlim(center[2] - half, center[2] + half)
            self.ax.legend(loc="upper right")
        self._apply(st); self._view = self._capture(); self.draw_idle()

class PdfMainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("分布载荷针体挠度仿真 (PDF 模型)")
        self.model = PdfEBBeamModel()
        self.canvas = NeedleCanvas()
        self.controls = {
            "L": ParamSlider("L 针长度", 50, 200, 1, 200 * 1000 if False else 200, "mm", precision=0),
            "Fx": ParamSlider("Fx 轴向力", -5, 15, 0.1, 0.0, "N", precision=1),
            "Fy": ParamSlider("Fy 横向力", -5, 5, 0.1, 0.0, "N", precision=1),
            "Fz": ParamSlider("Fz 横向力", -5, 5, 0.1, 0.0, "N", precision=1),
            "Mx": ParamSlider("Mx 力矩", -0.3, 0.3, 0.01, 0.0, "N·m", precision=2),
            "My": ParamSlider("My 力矩", -0.3, 0.3, 0.01, 0.0, "N·m", precision=2),
            "Mz": ParamSlider("Mz 力矩", -0.3, 0.3, 0.01, 0.0, "N·m", precision=2),
            "f_fric": ParamSlider("摩擦力 f_fric", 0, 5, 0.05, 0.0, "N", precision=2),
            "f_insert": ParamSlider("插入长度 f_insert", 0, 200, 1, 100, "mm", precision=0),
        }
        control_box = QtWidgets.QGroupBox("载荷与参数")
        v = QtWidgets.QVBoxLayout(control_box)
        for w in self.controls.values():
            v.addWidget(w); w.value_changed.connect(self.update_solution)
        self.metric_labels = {}
        metric_box = QtWidgets.QGroupBox("计算结果")
        form = QtWidgets.QFormLayout(metric_box)
        for key, title, fmt in [
            ("Ux", "Uₓ (mm)", "{:.3f}"), ("Uy", "Uᵧ (mm)", "{:.3f}"),
            ("Uz", "U_z (mm)", "{:.3f}"), ("Uyz", "Uyz (mm)", "{:.3f}"),
            ("theta_y", "θᵧ (rad)", "{:.4f}"), ("theta_z", "θ_z (rad)", "{:.4f}")
        ]:
            lbl = QtWidgets.QLabel(fmt.format(0.0)); self.metric_labels[key] = lbl
            form.addRow(QtWidgets.QLabel(title), lbl)
        self.status_label = QtWidgets.QLabel("等待输入…"); self.status_label.setStyleSheet("color:#22c55e;font-weight:bold;")
        form.addRow(QtWidgets.QLabel("状态"), self.status_label)
        right = QtWidgets.QVBoxLayout(); right.addWidget(self.canvas, stretch=3); right.addWidget(metric_box, stretch=1)
        main = QtWidgets.QHBoxLayout(self); main.addWidget(control_box, stretch=1); main.addLayout(right, stretch=2)
        self.update_solution()
    def _rebuild_model_if_needed(self, L_m):
        if abs(L_m - self.model.L) > 1e-6:
            self.model = PdfEBBeamModel(needle_length_m=L_m, N_grid=self.model.N)
    def update_solution(self):
        params = {k: c.value() for k, c in self.controls.items()}
        L_m = params["L"] / 1000.0
        self._rebuild_model_if_needed(L_m)
        self.model.update_user_inputs(params["f_fric"], params["f_insert"] / 1000.0)
        Fx, Fy, Fz, Mx, My, Mz = params["Fx"], params["Fy"], params["Fz"], params["Mx"], params["My"], params["Mz"]
        res = self.model.compute_deflection(Fx, Fy, Fz, Mx, My, Mz)
        for key, fmt in [("Ux", "{:.3f}"), ("Uy", "{:.3f}"), ("Uz", "{:.3f}"), ("Uyz", "{:.3f}"),
                         ("theta_y", "{:.4f}"), ("theta_z", "{:.4f}")]:
            val = res[key] * 1000 if key.startswith("U") else res[key]
            self.metric_labels[key].setText(fmt.format(val))
        self.status_label.setText("已更新"); self.status_label.setStyleSheet("color:#22c55e;font-weight:bold;")
        curve, baseline = compute_geometry_pdf(L_m, res)
        insert_start = self.model.e_out  # = L - f_insert
        self.canvas.update_scene(curve, baseline, insert_start_m=insert_start)

def run_pdf_ui():
    app = QtWidgets.QApplication(sys.argv)
    win = PdfMainWindow()
    win.resize(1100, 700)
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    print("Install dependencies with: pip install pyqt5 matplotlib numpy")
    run_pdf_ui()

