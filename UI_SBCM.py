print('Install dependencies with: pip install pyqt5 matplotlib numpy')
import sys
import numpy as np
from PyQt5 import QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # 保证 3D 支持


class NeedleDeflectionModel:
    """与 SBCM.py 中一致的针体挠度模型（Newton-Raphson 解法）"""

    def __init__(self):
        self.needle_length = 185e-3
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
            [-3.0 / 5, 1.0 / 20, 0.0, 0.0],
            [1.0 / 20, -1.0 / 15, 0.0, 0.0],
            [0.0, 0.0, -3.0 / 5, -1.0 / 20],
            [0.0, 0.0, -1.0 / 20, -1.0 / 15]
        ], dtype=float)

        self.H3 = np.array([
            [0.0, 0.0, 0.0, -0.5],
            [0.0, 0.0, -0.5, -0.25],
            [0.0, -0.5, 0.0, 0.0],
            [-0.5, -0.25, 0.0, 0.0]
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

    def normalize_parameters(self, F_x, F_y, F_z, M_x, M_y, M_z, L, n=1):
        denom = self.E * self.I
        L2 = L * L
        return (
            F_x * L2 / (n ** 2 * denom),
            F_y * L2 / (n ** 2 * denom),
            F_z * L2 / (n ** 2 * denom),
            M_x * L / (n * denom),
            M_y * L / (n * denom),
            M_z * L / (n * denom),
        )

    def denormalize_displacement(self, u_y, theta_z, u_z, theta_y, L, n=1):
        return (
            u_y * L / n,
            theta_z,
            u_z * L / n,
            theta_y
        )

    def calculate_residual(self, v, f_x, f_y, f_z, m_x, m_y, m_z):
        u_y, theta_z, u_z, theta_y = v
        m_xd = m_x + theta_z * m_y + theta_y * m_z
        g = np.array([f_y, m_z, f_z, m_y], dtype=float)

        term1 = self.H1 @ v
        term2_matrix = (2 * f_x) * self.H2 + m_xd * (2 * self.H3 + self.H7)
        term3_matrix = (
            (f_x ** 2) * self.H4 +
            (m_xd * f_z) * self.H5 +
            (m_xd ** 2) * self.H6
        )
        term2 = term2_matrix @ v
        term3 = term3_matrix @ v
        return g - (term1 - term2 - term3)

    def calculate_jacobian(self, v, f_x, f_y, f_z, m_x, m_y, m_z):
        u_y, theta_z, u_z, theta_y = v
        m_xd = m_x + theta_z * m_y + theta_y * m_z

        J = -self.H1.copy()
        J += (2 * f_x) * self.H2 + m_xd * (2 * self.H3 + self.H7)
        J += (f_x ** 2) * self.H4 + (m_xd * f_z) * self.H5 + (m_xd ** 2) * self.H6

        dm_xd_dv = np.array([0.0, m_y, 0.0, m_z], dtype=float)
        base_linear = 2 * self.H3 + self.H7
        term_linear = base_linear @ v
        term_quadratic = (f_z * self.H5 + 2 * m_xd * self.H6) @ v

        J += np.outer(term_linear, dm_xd_dv)
        J += np.outer(term_quadratic, dm_xd_dv)
        return J

    def calculate_tip_deflection(self, F_x, F_y, F_z, M_x, M_y, M_z, L):
        L = max(L, 1e-6)
        f_x, f_y, f_z, m_x, m_y, m_z = self.normalize_parameters(F_x, F_y, F_z, M_x, M_y, M_z, L)

        g = np.array([f_y, m_z, f_z, m_y], dtype=float)
        try:
            v = 0.5 * np.linalg.solve(self.H1, g)
        except np.linalg.LinAlgError:
            v = np.array([1e-3, 1e-3, 1e-3, 1e-3], dtype=float)

        max_iter = 20
        tol = 1e-8
        residual_norm = np.inf
        iterations = 0

        for i in range(max_iter):
            F_v = self.calculate_residual(v, f_x, f_y, f_z, m_x, m_y, m_z)
            residual_norm = np.linalg.norm(F_v)
            iterations = i + 1
            if residual_norm < tol:
                break

            J = self.calculate_jacobian(v, f_x, f_y, f_z, m_x, m_y, m_z)
            try:
                delta_v = np.linalg.solve(J, -F_v)
            except np.linalg.LinAlgError:
                delta_v, *_ = np.linalg.lstsq(J, -F_v, rcond=None)

            if np.linalg.norm(delta_v) > 10.0:
                delta_v *= 0.5

            v_new = v + delta_v
            if not np.all(np.isfinite(v_new)):
                v = np.array([1e-4, 1e-4, 1e-4, 1e-4])
                break
            v = v_new

        u_y, theta_z, u_z, theta_y = v
        U_y, Theta_z, U_z, Theta_y = self.denormalize_displacement(u_y, theta_z, u_z, theta_y, L)
        U_x = (F_x * L) / (self.E * self.A)

        return {
            "U_x": U_x,
            "U_y": U_y,
            "U_z": U_z,
            "Theta_y": Theta_y,
            "Theta_z": Theta_z,
            "iterations": iterations,
            "residual": residual_norm,
            "converged": residual_norm < tol,
            "length": L
        }


class ParamSlider(QtWidgets.QWidget):
    """封装带滑块的参数控件"""

    value_changed = QtCore.pyqtSignal(float)

    def __init__(self, title, min_val, max_val, step, initial, unit="", precision=2, parent=None):
        super().__init__(parent)
        self.min_val = min_val
        self.max_val = max_val
        self.step = step
        self.unit = unit
        self.precision = precision

        layout = QtWidgets.QVBoxLayout(self)
        label = QtWidgets.QLabel(f"{title} ({min_val} ~ {max_val} {unit})")
        layout.addWidget(label)

        slider_layout = QtWidgets.QHBoxLayout()
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        ticks = int(round((max_val - min_val) / step))
        self.slider.setMinimum(0)
        self.slider.setMaximum(ticks)
        start = int(round((initial - min_val) / step))
        self.slider.setValue(start)
        slider_layout.addWidget(self.slider)

        self.value_label = QtWidgets.QLabel()
        self.value_label.setFixedWidth(90)
        slider_layout.addWidget(self.value_label)
        layout.addLayout(slider_layout)

        self.slider.valueChanged.connect(self._emit_value)
        self._emit_value(self.slider.value())

    def _emit_value(self, slider_value):
        actual = self.min_val + slider_value * self.step
        fmt = f"{{:.{self.precision}f}}"
        text = fmt.format(actual)
        if self.unit:
            text += f" {self.unit}"
        self.value_label.setText(text)
        self.value_changed.emit(actual)

    def value(self):
        slider_value = self.slider.value()
        return self.min_val + slider_value * self.step


class NeedleCanvas(FigureCanvas):
    """封装 Matplotlib 3D 画布，保持用户视角"""

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(5, 4), tight_layout=True)
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.fig.patch.set_facecolor("#0f172a")
        self.ax.set_facecolor("#0b1120")
        self.ax.set_xlabel("X (mm)")
        self.ax.set_ylabel("Y (mm)")
        self.ax.set_zlabel("Z (mm)")
        self.ax.view_init(elev=22, azim=-60)
        self.view_state = self._capture_view()
        self._connect_interaction_listeners()

    def _capture_view(self):
        return {
            "elev": getattr(self.ax, "elev", 30.0),
            "azim": getattr(self.ax, "azim", -60.0),
            "dist": getattr(self.ax, "dist", 10.0)
        }

    def _apply_view(self, state):
        self.ax.view_init(elev=state.get("elev", 30.0), azim=state.get("azim", -60.0))
        dist = state.get("dist")
        if dist is not None:
            self.ax.dist = dist

    def _connect_interaction_listeners(self):
        events = ("button_release_event", "scroll_event", "key_release_event")
        for evt in events:
            self.fig.canvas.mpl_connect(evt, self._handle_view_change)

    def _handle_view_change(self, event):
        self.view_state = self._capture_view()

    def update_scene(self, curve, baseline):
        preserved_view = self.view_state.copy()
        self.ax.cla()
        self.ax.set_xlabel("X (mm)")
        self.ax.set_ylabel("Y (mm)")
        self.ax.set_zlabel("Z (mm)")

        if curve is not None and len(curve) > 0:
            self.ax.plot(baseline[:, 0], baseline[:, 1], baseline[:, 2],
                         color="#94a3b8", linestyle="--", linewidth=1.5, label="Baseline")
            self.ax.plot(curve[:, 0], curve[:, 1], curve[:, 2],
                         color="#38bdf8", linewidth=3, label="Needle")
            self.ax.scatter(curve[-1, 0], curve[-1, 1], curve[-1, 2],
                            color="#f97316", s=40, label="Tip")

            all_pts = np.vstack((curve, baseline))
            max_range = (all_pts.max(axis=0) - all_pts.min(axis=0)).max()
            centers = all_pts.mean(axis=0)
            half = max_range / 2 if max_range > 0 else 1
            self.ax.set_xlim(centers[0] - half, centers[0] + half)
            self.ax.set_ylim(centers[1] - half, centers[1] + half)
            self.ax.set_zlim(centers[2] - half, centers[2] + half)
            self.ax.legend(loc="upper right")

        self._apply_view(preserved_view)
        self.view_state = self._capture_view()
        self.draw_idle()


def cubic_hermite(t, end_value, start_slope, end_slope, L):
    """三次 Hermite 插值（内部用于生成弯曲曲线）"""
    t2 = t * t
    t3 = t2 * t
    h00 = 2 * t3 - 3 * t2 + 1
    h10 = t3 - 2 * t2 + t
    h01 = -2 * t3 + 3 * t2
    h11 = t3 - t2
    return h10 * (start_slope * L) + h01 * end_value + h11 * (end_slope * L)


def compute_geometry(L, result, segments=120):
    safe_L = max(L, 1e-6)
    s = np.linspace(0, safe_L, segments)
    t = s / safe_L
    y = cubic_hermite(t, result["U_y"], 0.0, result["Theta_z"], safe_L)
    z = cubic_hermite(t, result["U_z"], 0.0, -result["Theta_y"], safe_L)
    x = s + result["U_x"] * t
    curve = np.column_stack((x, y, z)) * 1000  # 转为 mm
    baseline = np.column_stack((s, np.zeros_like(s), np.zeros_like(s))) * 1000
    return curve, baseline


class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("针体挠度交互仿真 (Python)")
        self.model = NeedleDeflectionModel()

        self.canvas = NeedleCanvas()

        self.controls = {
            "L": ParamSlider("插入长度", 10, 185, 1, 150, "mm", precision=0),
            "Fx": ParamSlider("Fx 轴向力", -10, 10, 0.1, 0.0, "N", precision=1),
            "Fy": ParamSlider("Fy 横向力", -5, 5, 0.1, 0.0, "N", precision=1),
            "Fz": ParamSlider("Fz 横向力", -5, 5, 0.1, 0.0, "N", precision=1),
            "Mx": ParamSlider("Mx 力矩", -0.25, 0.25, 0.01, 0.0, "N·m", precision=2),
            "My": ParamSlider("My 力矩", -0.25, 0.25, 0.01, 0.0, "N·m", precision=2),
            "Mz": ParamSlider("Mz 力矩", -0.25, 0.25, 0.01, 0.0, "N·m", precision=2),
        }

        control_box = QtWidgets.QGroupBox("载荷与几何参数（拖动滑块）")
        control_layout = QtWidgets.QVBoxLayout(control_box)
        for widget in self.controls.values():
            control_layout.addWidget(widget)
            widget.value_changed.connect(self.update_solution)

        self.metric_labels = {}
        metric_box = QtWidgets.QGroupBox("计算结果")
        metric_layout = QtWidgets.QFormLayout(metric_box)
        metric_specs = [
            ("U_x", "Uₓ (mm)", "0.000"),
            ("U_y", "Uᵧ (mm)", "0.000"),
            ("U_z", "U_z (mm)", "0.000"),
            ("Theta_y", "θᵧ (rad)", "0.0000"),
            ("Theta_z", "θ_z (rad)", "0.0000"),
            ("iterations", "迭代次数", "0"),
            ("residual", "残差范数", "0"),
        ]
        for key, title, init in metric_specs:
            value_label = QtWidgets.QLabel(init)
            self.metric_labels[key] = value_label
            metric_layout.addRow(QtWidgets.QLabel(title), value_label)

        self.status_label = QtWidgets.QLabel("等待输入…")
        self.status_label.setStyleSheet("color: #22c55e; font-weight: bold;")
        metric_layout.addRow(QtWidgets.QLabel("状态"), self.status_label)

        right_layout = QtWidgets.QVBoxLayout()
        right_layout.addWidget(self.canvas, stretch=3)
        right_layout.addWidget(metric_box, stretch=1)

        main_layout = QtWidgets.QHBoxLayout(self)
        main_layout.addWidget(control_box, stretch=1)
        main_layout.addLayout(right_layout, stretch=2)

        self.update_solution()

    def update_solution(self):
        params = {key: ctrl.value() for key, ctrl in self.controls.items()}
        L_m = params["L"] / 1000.0
        Fx, Fy, Fz = params["Fx"], params["Fy"], params["Fz"]
        Mx, My, Mz = params["Mx"], params["My"], params["Mz"]

        result = self.model.calculate_tip_deflection(Fx, Fy, Fz, Mx, My, Mz, L_m)

        self.metric_labels["U_x"].setText(f"{result['U_x'] * 1000:.3f}")
        self.metric_labels["U_y"].setText(f"{result['U_y'] * 1000:.3f}")
        self.metric_labels["U_z"].setText(f"{result['U_z'] * 1000:.3f}")
        self.metric_labels["Theta_y"].setText(f"{result['Theta_y']:.4f}")
        self.metric_labels["Theta_z"].setText(f"{result['Theta_z']:.4f}")
        self.metric_labels["iterations"].setText(str(result["iterations"]))
        self.metric_labels["residual"].setText(f"{result['residual']:.2e}")

        status_text = "✅ 收敛" if result["converged"] else "⚠️ 未完全收敛"
        status_color = "#22c55e" if result["converged"] else "#f97316"
        self.status_label.setText(status_text)
        self.status_label.setStyleSheet(f"color: {status_color}; font-weight: bold;")

        curve, baseline = compute_geometry(L_m, result)
        self.canvas.update_scene(curve, baseline)


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.resize(1100, 700)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    print('Install dependencies with: pip install pyqt5 matplotlib numpy')
    main()