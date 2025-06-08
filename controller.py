import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R

class AdmittanceController:
    def __init__(self, model, data, site_name, M, B, K, Kp, Kd, desired_activation):
        self.model = model
        self.data = data
        self.M = M
        self.B = B
        self.K = K
        self.Kp = Kp
        self.Kd = Kd
        self.site_name = site_name
        self.desired_activation = desired_activation

        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)

        self.x = np.zeros(6)
        self.xd = np.zeros(6)
        self.xdd = np.zeros(6)
        self.external_force = np.zeros(6)

        self.traj_func = None
        self.vel_func = None

    def set_external_force(self, force,t):
        self.external_force = np.array(force) 

    def set_trajectory(self, traj_func, vel_func):
        self.traj_func = traj_func
        self.vel_func = vel_func

    def force_on_patient(self, pos_err, K): # TODO: connect this to the input of the OpenSim model
        F = pos_err @ K
        return F
    
    def set_K(self, simulated_activation, desired_activation):
        activation_err = desired_activation - simulated_activation # this is an arbitrary control law for changing K
        scale = (1 + 0.001* activation_err)
        K_new = self.K * scale
        return K_new
    
    def opensim_simulated_activation(self,t): #TODO: connect this to the output of the OpenSim model
        fake = 1.0 + 0.1*np.sin(t) # For Ben's debugging
        return fake
    
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

        self.simulated_activation = self.opensim_simulated_activation(t)

        self.K = self.set_K(self.simulated_activation, self.desired_activation)
        print('K is:')
        print(self.K)

        external_force = -self.force_on_patient(pos_err, self.K) # we are updating the external force applied to the simulation as well

        rhs = external_force - self.B @ vel_err - self.K @ pos_err
        self.xdd = np.linalg.solve(self.M, rhs)

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
