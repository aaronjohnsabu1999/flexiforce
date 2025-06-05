import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R

class AdmittanceController:
    def __init__(self, model, data, site_name, M, B, K, Kp, Kd):
        self.model = model
        self.data = data
        self.M = M
        self.B = B
        self.K = K
        self.Kp = Kp
        self.Kd = Kd
        self.site_name = site_name

        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)

        self.x = np.zeros(6)
        self.xd = np.zeros(6)
        self.xdd = np.zeros(6)
        # self.force_measured = np.zeros(6)
        self.external_force = np.zeros(6)

        self.traj_func = None
        self.vel_func = None

    def set_external_force(self, force,t):
        self.external_force = np.sin(t)*np.array(force) # TODO: ADD THE FORCE APPLIED by the EMG HERE

    def set_trajectory(self, traj_func, vel_func):
        self.traj_func = traj_func
        self.vel_func = vel_func

    def get_current_pose(self):
        pos = self.data.site_xpos[self.site_id].copy()
        mat = self.data.site_xmat[self.site_id].reshape((3, 3))
        quat = R.from_matrix(mat).as_quat()
        rotvec = R.from_quat(quat).as_rotvec()
        return pos, quat, np.hstack([pos, rotvec])

    def get_reference_at_time(self, t):
        x_ref = np.array(self.traj_func(t))
        xd_ref = np.array(self.vel_func(t))
        return x_ref, xd_ref

    def compute_torques(self, t, dt):
        mujoco.mj_forward(self.model, self.data)

        pos, quat, x_meas = self.get_current_pose()
        x_ref, xd_ref = self.get_reference_at_time(t)

        pos_err = x_meas - x_ref
        vel_err = self.xd - xd_ref

        rhs = self.external_force - self.B @ vel_err - self.K @ pos_err
        self.xdd = np.linalg.solve(self.M, rhs.reshape(-1, 1)).flatten()

        self.xd += self.xdd * dt
        self.x += self.xd * dt

        Jp = np.zeros((3, self.model.nv))
        Jr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, Jp, Jr, self.site_id)
        J = np.vstack((Jp, Jr))

        qvel_des = np.linalg.pinv(J) @ self.xd
        qvel_error = qvel_des - self.data.qvel
        tau = self.Kp * qvel_error - self.Kd * self.data.qvel

        return tau
