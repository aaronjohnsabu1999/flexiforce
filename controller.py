import numpy as np
import mujoco


class ParallelForceMotionController:
    def __init__(self, model, data, site_name, Kp=250.0, Kd=5.0):
        self.model = model
        self.data = data
        self.site_id = model.site(site_name).id
        self.Kp = Kp
        self.Kd = Kd
        self.force_z = -5.0  # default z-force

    def set_force_z(self, fz):
        self.force_z = fz

    def compute_torques(self, x_goal):
        mujoco.mj_forward(self.model, self.data)
        x_curr = self.data.site_xpos[self.site_id]
        dx = x_goal[:2] - x_curr[:2]  # xy control only

        J_pos = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, J_pos, None, self.site_id)

        F = np.zeros(3)
        F[:2] = self.Kp * dx
        F[2] = self.force_z

        tau = J_pos.T @ F - self.Kd * self.data.qvel
        return tau, F


class AdmittanceController:
    def __init__(self, model, data, site_name, M=1.0, B=50.0, K=0.0, Kp=100.0, Kd=0.0):
        self.model = model
        self.data = data
        self.site_id = model.site(site_name).id

        # Admittance parameters
        self.M = M
        self.B = B
        self.K = K

        # Velocity control gains
        self.Kp = Kp
        self.Kd = Kd

        # State
        self.xd = np.zeros(3)
        self.x = None
        self.prev_time = None
        self.f_ext = np.zeros(3)

    def set_external_force(self, f_ext):
        self.f_ext = np.array(f_ext)

    def compute_torques(self, dt):
        mujoco.mj_forward(self.model, self.data)
        x_now = self.data.site_xpos[self.site_id]

        if self.x is None:
            self.x = x_now.copy()

        # Admittance dynamics: M ẍ + B ẋ + K x = F_ext
        try:
            # Clamp internal values
            force_term = self.f_ext
            damping_term = np.clip(self.B * self.xd, -1000.0, 1000.0)
            spring_term = np.clip(self.K * (self.x - x_now), -1000.0, 1000.0)
            
            acc = (force_term - damping_term - spring_term) / max(self.M, 1e-4)
            acc = np.clip(acc, -100.0, 100.0)  # prevent runaway integration
        except Exception as e:
            print(f"Admittance error: {e}")
            acc = np.zeros(3)

        self.xd += acc * dt
        self.x += self.xd * dt

        # Safety: reset if velocity explodes
        if np.any(np.abs(self.xd) > 1000.0):
            print("⚠️ Warning: xd overflow — resetting to zero.")
            self.xd[:] = 0.0

        # Compute Jacobian
        J_pos = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, J_pos, None, self.site_id)

        # Map Cartesian velocity to joint velocity
        qvel_desired = np.linalg.pinv(J_pos) @ self.xd

        # Velocity-level control in joint space
        vel_error = qvel_desired - self.data.qvel
        tau = self.Kp * vel_error - self.Kd * self.data.qvel

        # Final safety clamp
        tau = np.nan_to_num(tau, nan=0.0, posinf=1000.0, neginf=-1000.0)

        return tau, self.f_ext
