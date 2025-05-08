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

