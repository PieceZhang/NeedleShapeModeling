import numpy as np
class NeedleDeflectionModel:
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
            [12.0,-6.0,0.0,0.0],
            [-6.0,4.0,0.0,0.0],
            [0.0,0.0,12.0,6.0],
            [0.0,0.0,6.0,4.0]
        ], dtype=float)

        self.H2 = np.array([
            [-3.0/5, 1.0/20,0.0,0.0],
            [1.0/20,-1.0/15,0.0,0.0],
            [0.0,0.0,-3.0/5,-1.0/20],
            [0.0,0.0,-1.0/20,-1.0/15]
        ], dtype=float)

        self.H3 = np.array([
            [0.0,0.0,0.0,-1.0/2],
            [0.0,0.0,-1.0/2,-1.0/4],
            [0.0,-1.0/2,0.,0.0],
            [-1.0/2,-1.0/4,0.0,0.0]
        ], dtype=float)

        self.H4 = np.array([
            [1.0/700,-1.0/1400,0.0,0.0],
            [-1.0/1400,11.0/6300,0.0,0.0],
            [0.0,0.0,1.0/700,1.0/1400],
            [0.0,0.0,1.0/1400,11.0/6300]
        ], dtype=float)

        self.H5 = np.array([
            [0.0,0.0,0.0,1.0/60],
            [0.0,0.0,1.0/60,0.0],
            [0.0,1.0/60,0.0,0.0],
            [1.0/60,0.0,0.0,0.0]
        ], dtype=float)

        self.H6 = np.array([
            [1.0/5,-1.0/10,0.0,0.0],
            [-1.0/10,1.0/20,0.0,0.0],
            [0.0,0.0,1.0/5,1.0/10],
            [0.0,0.0,1.0/10,1.0/20]
        ], dtype=float)

        self.H7 = np.array([
            [0.0,0.0,0.0,0.0],
            [0.0,0.0,0.0,1.0],
            [0.0,0.0,0.0,0.0],
            [0.0,0.0,0.0,0.0]
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

    # g = H1*v - (2f_xH2 + m_xd(2H3+H7))*v - (f_x^2*H4 + m_xd*f_z*H5 + m_xd^2*H6)*v
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
        except:
            v = np.array([0.001, 0.001, 0.001, 0.001])

        max_iterations = 20
        tolerance = 1e-8
        for i in range(max_iterations):
            F_v = self.calculate_residual(v, f_x, f_y, f_z, m_x, m_y, m_z)
            J = self.calculate_jacobian(v, f_x, f_y, f_z, m_x, m_y, m_z)

            residual_norm = np.linalg.norm(F_v)
            if residual_norm < tolerance:
                break
            try:
                delta_v = np.linalg.solve(J, -F_v)
            except np.linalg.LinAlgError:
                delta_v = np.linalg.lstsq(J, -F_v, rcond=None)[0]
            v_old = v.copy()
            v = v + delta_v
            update_norm = np.linalg.norm(delta_v)
            if update_norm > 10.0: 
                         v = v_old + 0.5 * delta_v
            if np.any(np.isnan(v)) or np.any(np.isinf(v)):
                v = np.array([0.0001, 0.0001, 0.0001, 0.0001])
        u_y, theta_z, u_z, theta_y = v
        U_y, Theta_z, U_z, Theta_y = self.denormalize_displacement(u_y, theta_z, u_z, theta_y, L)
        U_x = (F_x * L) / (self.E * self.A)
        print(f"Final deflection results:")
        print(f"  U_x = {U_x * 1000:.3f} mm, U_y = {U_y * 1000:.3f} mm, U_z = {U_z * 1000:.3f} mm")
        print(f"  θ_y = {Theta_y:.3f} rad, θ_z = {Theta_z:.3f} rad")
        return U_x, U_y, U_z, Theta_y, Theta_z

def get_user_input():
    insertion_length = float(input("Enter insertion length (mm) - distance from needle tip to tissue surface: "))
    print("\nEnter force values (N):")
    Fx = float(input("Fx (axial force): "))
    Fy = float(input("Fy (lateral force in Y direction): "))
    Fz = float(input("Fz (lateral force in Z direction): "))
    print("\nEnter moment values (N·m):")
    Mx = float(input("Mx (moment about X axis): "))
    My = float(input("My (moment about Y axis): "))
    Mz = float(input("Mz (moment about Z axis): "))
    sensor_data = {
        'Fx': Fx, 'Fy': Fy, 'Fz': Fz,
        'Mx': Mx, 'My': My, 'Mz': Mz
    }
    return insertion_length, sensor_data

def main():
    model = NeedleDeflectionModel()
    insertion_length, sensor_data = get_user_input()
    U_x, U_y, U_z, Theta_y, Theta_z = model.calculate_tip_deflection_newton(
        sensor_data['Fx'], sensor_data['Fy'], sensor_data['Fz'],
        sensor_data['Mx'], sensor_data['My'], sensor_data['Mz'],
        model.needle_length
    )
    return U_x, U_y, U_z, Theta_y, Theta_z, insertion_length, sensor_data

if __name__ == "__main__":
    U_x, U_y, U_z, Theta_y, Theta_z, insertion_length, sensor_data = main()