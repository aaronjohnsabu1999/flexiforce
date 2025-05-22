from scipy.spatial.transform import Rotation as R
import numpy as np
import mujoco

class AdmittanceController:
    def __init__(self, model, data, site_name, *args, **kwargs):
        self.model = model
        self.data = data
        self.site_name = site_name
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        self.verbose = kwargs.get("verbose", False)

        # Admittance parameters
        M_val = kwargs.get("M", 1.0)
        B_val = kwargs.get("B", 50.0)
        K_val = kwargs.get("K", 10.0)
        self.M = np.ones(6) * M_val
        self.B = np.ones(6) * B_val
        self.K = np.ones(6) * K_val

        # Velocity control gains
        self.Kp = kwargs.get("Kp", 100.0)
        self.Kd = kwargs.get("Kd", 0.0)

        # State
        self.x = None           # current pose [xyz + rotvec]
        self.xd = np.zeros(6)   # twist: [vx, vy, vz, wx, wy, wz]
        self.force = np.zeros(6)

        # Trajectory
        self.initial_pose = None
        self.ref_vel = np.array([0.4, 0.0, -0.01, 0, 0, 0])       # move down in Z
        self.ref_offset = np.array([0.0, 0.0, -0.01, 0, 0, 0])      # 20 cm in front

    def set_force(self, force, **kwargs):
        assert force.shape == (6,), f"Force must be 6D, got {force.shape}"
        self.force = force

    def compute_torques(self, dt, time):
        mujoco.mj_forward(self.model, self.data)

        # Get current site pose
        p_now = self.data.site_xpos[self.site_id].copy()
        R_now = self.data.site_xmat[self.site_id].reshape((3, 3))
        rotvec_now = R.from_matrix(R_now).as_rotvec()
        x_now = np.hstack((p_now, rotvec_now))  # [6]

        # Initialize on first call
        if self.initial_pose is None:
            self.initial_pose = x_now + self.ref_offset
        if self.x is None:
            self.x = self.initial_pose.copy()
            self.xd = np.zeros(6)

        # Time-varying desired trajectory
        x_ref = self.initial_pose + self.ref_vel * time

        # Admittance dynamics: M ẍ + B ẋ + K (x - x_ref) = F
        acc = (self.force - self.B * self.xd - self.K * (self.x - x_ref)) / self.M
        self.xd += acc * dt
        self.x += self.xd * dt

        # Compute full spatial Jacobian
        Jp = np.zeros((3, self.model.nv))
        Jr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, Jp, Jr, self.site_id)
        J = np.vstack((Jp, Jr))  # shape (6, nv)

        # Desired joint velocity via pseudo-inverse
        qvel_desired = np.linalg.pinv(J) @ self.xd

        # Velocity-level PD control
        vel_error = qvel_desired - self.data.qvel
        tau = self.Kp * vel_error - self.Kd * self.data.qvel

        if self.verbose:
            print(f"[AC] t={time:.2f} | x_ref: {x_ref[:3]} | force_z={self.force[2]:.2f} | ||tau||={np.linalg.norm(tau):.2f}")

        return tau, self.force
