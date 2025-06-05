import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

class AdmittanceController:
    def __init__(self, model, data, site_name, **kwargs):
        self.model = model
        self.data = data
        self.site_name = site_name
        
        # Find site ID, create if doesn't exist
        try:
            self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        except:
            print(f"Warning: Site '{site_name}' not found. Using end-effector approximation.")
            self.site_id = 0  # Use first site as fallback
            
        self.verbose = kwargs.get("verbose", False)
        
        # Admittance parameters (6DOF: [fx, fy, fz, mx, my, mz])
        self.M = np.diag(np.ones(6) * kwargs.get("M", 1.0))  # Virtual mass/inertia
        print('The M Matrix is:')
        print(self.M)
        
        self.B = np.diag(np.ones(6) * kwargs.get("B", 50.0))  # Virtual damping
        print('The B Matrix is:')
        print(self.B)
        
        self.K = np.diag(np.ones(6) * kwargs.get("K", 10.0))  # Virtual stiffness
        print('The K Matrix is:')
        print(self.K)
        
        # Control gains for joint-level control
        self.Kp = kwargs.get("Kp", 100.0)
        print('The K_p Matrix is:')
        print(self.Kp)

        self.Kd = kwargs.get("Kd", 10.0)
        print('The K_d Matrix is:')
        print(self.Kd)
        
        # State variables for admittance dynamics
        self.x_adm = None           # Admittance pose [xyz + rotation vector]
        self.xd_adm = np.zeros(6)   # Admittance velocity [v_trans, w_rot]
        self.xdd_adm = np.zeros(6)  # Admittance acceleration
        self.force_measured = np.zeros(6)
        self.force_desired = np.zeros(6)
        
        # Reference trajectory (time-variant)
        self.trajectory_func = None
        self.trajectory_velocity_func = None
        
        # Contact force sensing
        self.use_contact_forces = kwargs.get("use_contact_forces", False)
        self.contact_geom_ids = kwargs.get("contact_geom_ids", [])
        
        if self.verbose:
            print(f"[AC] Initialized with site_id={self.site_id}")
    
    def set_desired_force(self, force_desired):
        """Set the desired interaction force (6DOF)"""
        assert len(force_desired) == 6, f"Force must be 6D, got {len(force_desired)}"
        self.force_desired = np.array(force_desired).copy()
        if self.verbose:
            print(f"[AC] Set desired force: {self.force_desired}")
    
    def set_reference_trajectory(self, trajectory_func, velocity_func=None):
        """Set time-variant reference trajectory
        
        Args:
            trajectory_func: Function that takes time t and returns 6D pose [x,y,z,rx,ry,rz]
            velocity_func: Function that takes time t and returns 6D velocity (optional)
        """
        self.trajectory_func = trajectory_func
        self.trajectory_velocity_func = velocity_func
        if self.verbose:
            print(f"[AC] Set time-variant reference trajectory")
    
    def get_reference_at_time(self, t):
        """Get reference pose and velocity at given time"""
        if self.trajectory_func is None:
            # Default stationary trajectory at current position
            if self.x_adm is None:
                self.x_adm = x_ref.copy()
                self.xd_adm = xd_ref.copy()

            else:
                return np.zeros(6), np.zeros(6)
        
        x_ref = np.array(self.trajectory_func(t))
        
        if self.trajectory_velocity_func is not None:
            xd_ref = np.array(self.trajectory_velocity_func(t))
        else:
            # Numerical differentiation for velocity
            dt = 1e-4
            x_ref_next = np.array(self.trajectory_func(t + dt))
            xd_ref = (x_ref_next - x_ref) / dt
        
        return x_ref, xd_ref
    
    
    def pose_to_spatial(self, pos, quat):
        """Convert position and quaternion to spatial pose vector"""
        # Use rotation vector for spatial representation
        rot_vec = R.from_quat(quat).as_rotvec()
        return np.hstack((pos, rot_vec))
    
    def get_current_pose(self):
        """Get current end-effector pose"""
        # Handle case where site doesn't exist
        try:
            pos = self.data.site_xpos[self.site_id].copy()
            rot_mat = self.data.site_xmat[self.site_id].reshape((3, 3))
        except:
            # Fallback: use end-effector body position
            if self.model.nbody > 1:
                body_id = self.model.nbody - 1  # Last body (usually end-effector)
                pos = self.data.xpos[body_id].copy()
                rot_mat = self.data.xmat[body_id].reshape((3, 3))
            else:
                # Ultimate fallback: use origin
                pos = np.zeros(3)
                rot_mat = np.eye(3)
        
        quat = R.from_matrix(rot_mat).as_quat()
        return pos, quat
    
    def compute_spatial_jacobian(self):
        """Compute 6DOF spatial Jacobian"""
        Jp = np.zeros((3, self.model.nv))
        Jr = np.zeros((3, self.model.nv))
        
        try:
            mujoco.mj_jacSite(self.model, self.data, Jp, Jr, self.site_id)
        except:
            # Fallback: use body Jacobian for last body
            if self.model.nbody > 1:
                body_id = self.model.nbody - 1
                mujoco.mj_jacBody(self.model, self.data, Jp, Jr, body_id)
            else:
                # If that fails, create identity Jacobian
                min_dim = min(3, self.model.nv)
                Jp[:min_dim, :min_dim] = np.eye(min_dim)
                Jr[:min_dim, :min_dim] = np.eye(min_dim)
        
        return np.vstack((Jp, Jr))  # Shape: (6, nv)
    
    def compute_torques(self, dt, time=0.0, **kwargs):
        mujoco.mj_forward(self.model, self.data)

        pos_now, quat_now = self.get_current_pose()
        x_actual = self.pose_to_spatial(pos_now, quat_now)

        if self.x_adm is None:
            self.x_adm = x_actual.copy()
            self.xd_adm = np.zeros(6)

        x_ref, xd_ref = self.get_reference_at_time(time)


        force_error = self.force_desired - self.force_measured

        position_error = self.x_adm - x_ref
        velocity_error = self.xd_adm - xd_ref

        M_safe = np.maximum(self.M, 1e-6)

        self.xdd_adm = (force_error - self.B @ self.xd_adm - self.K @ position_error) / M_safe

        self.xd_adm += self.xdd_adm * dt
        self.x_adm += self.xd_adm * dt

        J = self.compute_spatial_jacobian()
        


        if J.shape[1] > 0:
            J_pinv = np.linalg.pinv(J)
            qvel_desired = J_pinv @ self.xd_adm

            if len(qvel_desired) > self.model.nv:
                qvel_desired = qvel_desired[:self.model.nv]
            elif len(qvel_desired) < self.model.nv:
                temp = np.zeros(self.model.nv)
                temp[:len(qvel_desired)] = qvel_desired
                qvel_desired = temp
        else:
            qvel_desired = np.zeros(self.model.nv)

        qvel_error = qvel_desired - self.data.qvel
        tau = self.Kp * qvel_error - self.Kd * self.data.qvel


        if self.verbose and time % 1.0 < dt:
            print(f"[AC] t={time:.2f} | "
                f"F_err: [{force_error[0]:.1f}, {force_error[1]:.1f}, {force_error[2]:.1f}] | "
                f"||tau||: {np.linalg.norm(tau):.2f} | "
                f"x_adm: [{self.x_adm[0]:.3f}, {self.x_adm[1]:.3f}, {self.x_adm[2]:.3f}]"
                f"||J||: {np.linalg.norm(J):.4e}, rank: {np.linalg.matrix_rank(J)}")

        return tau, self.force_measured

    def reset(self):
        """Reset controller state"""
        self.x_adm = None
        self.xd_adm = np.zeros(6)
        self.xdd_adm = np.zeros(6)
        self.force_measured = np.zeros(6)
    
    def plot_results(self, log):
        """Plot simulation results"""
        if len(log["time"]) == 0:
            print("No data to plot!")
            return
            
        plt.figure(figsize=(15, 10))
        
        # Force plot
        plt.subplot(4, 1, 1)
        plt.plot(log["time"], log["force"], 'b-', linewidth=2, label="Measured Z Force [N]")
        if hasattr(self, 'force_desired') and len(self.force_desired) > 0:
            desired_force_z = [self.force_desired[2]] * len(log["time"])
            plt.plot(log["time"], desired_force_z, 'r--', linewidth=1, alpha=0.7, label="Desired Z Force")
        plt.ylabel("Z Force (N)")
        plt.grid(True)
        plt.legend()
        plt.title("Admittance Controller Results")
        
        # Position tracking
        plt.subplot(4, 1, 2)
        labels = ['X', 'Y', 'Z']
        for i in range(3):
            plt.plot(log["time"], log["position"][:, i], label=f'{labels[i]} Admittance')
            plt.plot(log["time"], log["measured_position"][:, i], '--', label=f'{labels[i]} Measured')
            plt.plot(log["time"], log["x_ref"][:, i], ':', label=f'{labels[i]} Ref')

        plt.ylabel("Position (m)")
        plt.grid(True)
        plt.legend(loc='upper right', ncol=3)
        plt.title("End-Effector Position: Admittance vs Measured vs Reference")

        # Joint velocities
        plt.subplot(4, 1, 3)
        if len(log["vel"]) > 0:
            for j in range(min(7, log["vel"].shape[1])):  # Show up to 7 joints
                plt.plot(log["time"], log["vel"][:, j], label=f'Joint {j+1}')
        plt.ylabel("Joint Velocity (rad/s)")
        plt.grid(True)
        plt.legend(loc="upper right")
        
        # Force error over time
        plt.subplot(4, 1, 4)
        if hasattr(self, 'force_desired') and len(self.force_desired) > 0:
            force_error = self.force_desired[2] - np.array(log["force"])
            plt.plot(log["time"], force_error, 'g-', linewidth=1, label="Force Error")
            plt.ylabel("Force Error (N)")
            plt.xlabel("Time (s)")
            plt.grid(True)
            plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        # Print summary statistics
        if len(log["time"]) > 0:
            print(f"\nSimulation Summary:")
            print(f"  Duration: {log['time'][-1]:.2f} seconds")
            print(f"  Final Z Force: {log['force'][-1]:.2f} N")
            print(f"  Max |Force|: {np.max(np.abs(log['force'])):.2f} N")
            if len(log["vel"]) > 0:
                print(f"  Max Joint Velocity: {np.max(np.abs(log['vel'])):.2f} rad/s")
