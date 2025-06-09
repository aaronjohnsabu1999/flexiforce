import numpy as np
import mujoco
from Bicep import Bicep_Curl
from scipy.spatial.transform import Rotation as R

class AdmittanceController:
    def __init__(self, model, data, site_name, M, B, K, Kp, Kd, desired_activation, sps = 100, curl_time = 10):
        self.model = model
        self.data = data
        self.M = M
        self.B = B
        self.K = K
        self.Kp = Kp
        self.Kd = Kd
        self.site_name = site_name
        self.desired_activation = desired_activation
        self.bicep = Bicep_Curl(sps = sps, curl_time = curl_time)
        self.t, _ = self.bicep._traj()
        self.dt = 1/sps

        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)

        self.x = np.zeros(6)
        self.xd = np.zeros(6)
        self.xdd = np.zeros(6)
        self.external_force = np.zeros(6)

        self.traj_func = None
        self.vel_func = None

    def set_external_force(self, force):
        self.external_force = np.array(force) 

    def set_trajectory(self, traj_func, vel_func):
        self.traj_func = traj_func
        self.vel_func = vel_func

    def force_on_patient(self, pos_err, K): 
        F = -pos_err @ K
        return F
    
    def set_K(self, simulated_activation, desired_activation, G):
        activation_err = desired_activation - simulated_activation # this is an arbitrary control law for changing K
        scale = (1 + G["G_p"]* activation_err + G["G_d"]*activation_err/self.dt)
        K_new = self.K * scale
        return K_new
    
    def opensim_simulated_activation(self, index, F): 
        return self.bicep.step_simulation(index, F)
    
    def get_current_pose(self):
        pos = self.data.site_xpos[self.site_id].copy()
        mat = self.data.site_xmat[self.site_id].reshape((3, 3))
        quat = R.from_matrix(mat).as_quat()
        rotvec = R.from_quat(quat).as_rotvec()
        return pos, quat, np.hstack([pos, rotvec])

    def get_reference_at_time(self, index):
        t = self.t[index]
        x_ref = np.array(self.traj_func(t))
        xd_ref = np.array(self.vel_func(t))
        return x_ref, xd_ref

    def compute_torques(self, index, G):

        mujoco.mj_forward(self.model, self.data)

        pos, quat, x_meas = self.get_current_pose()
        x_ref, xd_ref = self.get_reference_at_time(index)

        pos_err = x_meas - x_ref
        vel_err = self.xd - xd_ref

        if index == 0: 
            self.simulated_activation = 0
        
        self.K = self.set_K(self.simulated_activation, self.desired_activation, G)

        external_force = self.force_on_patient(pos_err, self.K) # we are updating the external force applied to the simulation as well
        self.simulated_activation = self.opensim_simulated_activation(index, external_force)[0]

        rhs = external_force - self.B @ vel_err - self.K @ pos_err
        self.xdd = np.linalg.solve(self.M, rhs)

        self.xd += self.xdd * self.dt
        self.x += self.xd * self.dt

        Jp = np.zeros((3, self.model.nv))
        Jr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, Jp, Jr, self.site_id)
        J = np.vstack((Jp, Jr))

        qvel_des = np.linalg.pinv(J) @ self.xd
        qvel_error = qvel_des - self.data.qvel
        tau = self.Kp * qvel_error - self.Kd * self.data.qvel

        return tau
    
    def _t(self):
        return self.t
