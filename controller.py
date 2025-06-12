import numpy as np
import mujoco
from bicep import Bicep_Curl
from scipy.spatial.transform import Rotation as R
import opensim as osim

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
        self.activation_err = [desired_activation]

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
        '''
        Force Provided in Mujoco coordinate system (Z is up)
        '''
        #because of the way we're setting K, need to make pos_err be a scalar, not a vector
        pos_mag = np.sqrt(pos_err[0]**2 + pos_err[1]**2 + pos_err[2]**2)
        F = -pos_mag * K  #F = -kx
        return F
    
    def set_K(self, simulated_activation, desired_activation, index, G):
        '''
        Set anisotropic stiffness in the Mujoco coordinate system (Z is up)
        '''

        self.activation_err.append(desired_activation - simulated_activation) # this is an arbitrary control law for changing K
        PID = G["G_p"]*self.activation_err[-1] + G["G_d"]*(self.activation_err[-2] - self.activation_err[-1])/self.dt + G["G_i"]*sum(self.activation_err)

        norm = self.bicep.hand_norm(index)
        normalizer = (norm[0]**2 + norm[1]**2 + norm[2]**2)**0.5
        K_norm = np.asarray([norm[0], norm[2], norm[1], 0, 0, 0])/normalizer
       
        return K_norm*PID
    
    def opensim_simulated_activation(self, index, F): 
        '''
        F provided in the Mujoco coordinate system (Z is up)
        '''
        return self.bicep.step_simulation(index, F)
    
    def get_current_pose(self):
        pos = self.data.site_xpos[self.site_id].copy()
        mat = self.data.site_xmat[self.site_id].reshape((3, 3))
        quat = R.from_matrix(mat).as_quat()
        rotvec = R.from_quat(quat).as_rotvec()
        return pos, quat, np.hstack([pos, rotvec])

    def get_reference_at_time(self, index):
        t = self.t[index]
        
        pos = osim.TimeSeriesTable(self.bicep.force_path_sto)
        x = pos.getDependentColumn("r_ulna_radius_hand_force_px")
        y = pos.getDependentColumn("r_ulna_radius_hand_force_py")
        z = pos.getDependentColumn("r_ulna_radius_hand_force_pz")

        x_ref = np.array([x[index] + 0.2, z[index] +0.1, y[index] - 0.2, 0, 0, 0]) #z&y are flipped bcuz opensim flips them for some godforsaken reason
        xd_ref = np.array(self.vel_func(t))

        return x_ref, xd_ref

    def compute_torques(self, index, G):

        mujoco.mj_forward(self.model, self.data)

        if index == 0: 
            self.simulated_activation = 0
        
        if index == len(self.t) - 1:
            x_des, xd_ref = self.get_reference_at_time(index)
        else:
            x_des, xd_ref = self.get_reference_at_time(index + 1) #get reference pose from next time step   

        _, _, self.x = self.get_current_pose()
        pos_err = x_des - self.x #x_ref is the pose at the next time step
        vel_err = self.xd - xd_ref
        
        #This calculates the K at the current time step from the activation at the previous time step
        self.K = self.set_K(self.simulated_activation, self.desired_activation, index, G)
        
        #Based on the error between the current pose and the desired pose at the next time step and K at this time step, calculate the force on the patient at this time step
        self.external_force = self.force_on_patient(pos_err, self.K) 

        #Based on the force applied to the patient at this time step, what will the activation at this time step be?
        self.simulated_activation = self.opensim_simulated_activation(index, self.external_force)[0]

        rhs = self.external_force - self.B @ vel_err - self.K @ pos_err
        self.xdd = np.linalg.solve(self.M, rhs)

        self.xd += self.xdd * self.dt

        Jp = np.zeros((3, self.model.nv))
        Jr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, Jp, Jr, self.site_id)
        J = np.vstack((Jp, Jr))

        qvel_des = np.linalg.pinv(J) @ self.xd
        qvel_error = qvel_des - self.data.qvel
        
        qpos_ik, success = self.inverse_kinematics(
            self.model, self.data,
            site_name=self.site_name,
            target_pos=x_des[:3],       
            target_quat=[0,0,1,0], # contstraining the orientation to be 
            tol=1e-4
        )

        q_err = qpos_ik - self.data.qpos

        tau = self.Kp * q_err + self.Kd * qvel_error
     
        return tau
    
    def _t(self):
        return self.t
    
    # This is the Inverse Kinematics method from Google Deepmind for calculating IK on MuJoCo models: https://github.com/google-deepmind/dm_control/blob/main/dm_control/utils/inverse_kinematics.py
    def nullspace_method(self, jac_joints, delta, regularization_strength=0.0):
        hess_approx = jac_joints.T @ jac_joints
        joint_delta = jac_joints.T @ delta
        if regularization_strength > 0:
            hess_approx += np.eye(hess_approx.shape[0]) * regularization_strength
            return np.linalg.solve(hess_approx, joint_delta)
        else:
            return np.linalg.lstsq(hess_approx, joint_delta, rcond=-1)[0]

        
    def inverse_kinematics(self, model, data, site_name, target_pos=None, target_quat=None,
                       tol=1e-5, rot_weight=10.0, max_steps=100, max_update_norm=5.0,
                       reg_threshold=0.1, reg_strength=3e-2, joint_names=None):
        if target_pos is None and target_quat is None:
            raise ValueError("Must provide at least target_pos or target_quat.")

        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        update_nv = np.zeros(model.nv)
        jac = np.zeros((6, model.nv))
        err = np.zeros(6)

        for step in range(max_steps):
            mujoco.mj_fwdPosition(model, data)
            site_pos = data.site_xpos[site_id]
            site_mat = data.site_xmat[site_id].reshape(3, 3)

            err[:3] = target_pos[:3] - site_pos if target_pos is not None else 0
            err_norm = np.linalg.norm(err[:3])

            if target_quat is not None:
                from scipy.spatial.transform import Rotation as R
                current_quat = R.from_matrix(site_mat).as_quat()
                desired_quat = R.from_quat(target_quat)
                delta_rot = desired_quat * R.from_quat(current_quat).inv()
                err[3:] = delta_rot.as_rotvec()
                err_norm += np.linalg.norm(err[3:]) * rot_weight

            if err_norm < tol:
                return data.qpos.copy(), True

            Jp = np.zeros((3, model.nv))
            Jr = np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, Jp, Jr, site_id)
            jac[:3] = Jp
            jac[3:] = Jr if target_quat is not None else 0

            jac_joints = jac
            if joint_names is not None:
                raise NotImplementedError("Joint selection not implemented yet.")

            reg = reg_strength if err_norm > reg_threshold else 0.0
            dq = self.nullspace_method(jac_joints, err, regularization_strength=reg)
            dq_norm = np.linalg.norm(dq)
            if dq_norm > max_update_norm:
                dq *= max_update_norm / dq_norm

            mujoco.mj_integratePos(model, data.qpos, dq, 1)

            # Clamp to joint limits
            for i in range(model.nq):
                if model.jnt_limited[i]:
                    joint_id = model.jnt_qposadr[i]
                    qmin = model.jnt_range[i][0]
                    qmax = model.jnt_range[i][1]
                    data.qpos[joint_id] = np.clip(data.qpos[joint_id], qmin, qmax)

        return data.qpos.copy(), False
