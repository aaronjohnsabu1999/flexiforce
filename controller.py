from scipy.spatial.transform import Rotation as R
import numpy as np
import mujoco


class Controller:
    def __init__(self, model, data, site_name, verbose=False, *args, **kwargs):
        self.type = None
        self.model = model
        self.data = data
        self.site_name = site_name
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        self.verbose = kwargs.get("verbose", False)

    def set_force(self, force, target_mvc=None):
        assert force.shape == (6,), f"Force must be 6D, got {force.shape}"
        self.force = np.array(force)
        if target_mvc is not None:
            self.target_mvc = target_mvc

    def compute_torques(self, dt=None, **kwargs):
        raise NotImplementedError("Subclasses must implement compute_torques")

    def _log(self, message):
        if self.verbose:
            print(message)


class ParallelForceMotionController(Controller):
    def __init__(self, model, data, site_name, **kwargs):
        super().__init__(model, data, site_name, **kwargs)
        self.type = "PFC"
        self.Kp = kwargs.get("Kp", 250.0)
        self.Kd = kwargs.get("Kd", 5.0)

    def compute_torques(self, dt=None, **kwargs):
        x_goal = kwargs.get("x_goal", None)
        if dt is not None:
            self.dt = dt

        mujoco.mj_forward(self.model, self.data)
        x_curr = self.data.site_xpos[self.site_id]
        dx = x_goal[:2] - x_curr[:2]

        J_pos = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, J_pos, None, self.site_id)

        F = np.zeros(3)
        F[:2] = self.Kp * dx
        F[2] = self.force[2]

        tau = J_pos.T @ F - self.Kd * self.data.qvel
        self._log(
            f"[PFC] dx: {dx}, force: {self.force}, tau norm: {np.linalg.norm(tau):.2f}"
        )
        return tau, F


class AdmittanceController(Controller):
    def __init__(self, model, data, site_name, *args, **kwargs):
        verbose = kwargs.get("verbose", False)
        super().__init__(model, data, site_name, verbose=verbose)
        self.type = "AC"

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

        # Adaptation parameters
        self.adapt_gain_m = kwargs.get("adapt_gain_m", 0.0)
        self.adapt_gain_b = kwargs.get("adapt_gain_b", 0.0)
        self.adapt_gain_k = kwargs.get("adapt_gain_k", 0.0)

        # Admittance limits
        self.M_min = kwargs.get("M_min", None)
        self.M_max = kwargs.get("M_max", None)
        self.B_min = kwargs.get("B_min", None)
        self.B_max = kwargs.get("B_max", None)
        self.K_min = kwargs.get("K_min", None)
        self.K_max = kwargs.get("K_max", None)

        # State
        self.x = None  # current pose [xyz + rotvec]
        self.xd = np.zeros(6)  # twist: [vx, vy, vz, wx, wy, wz]
        self.force = np.zeros(6)

        # Trajectory
        self.initial_pose = None
        self.ref_vel = np.array([0.4, 0.0, -0.01, 0, 0, 0])  # move down in Z
        self.ref_offset = np.array([0.0, 0.0, -0.01, 0, 0, 0])  # 20 cm in front

    def compute_torques(self, dt=None, **kwargs):
        t = kwargs.get("t", 0.0)
        if dt is not None:
            self.dt = dt

        measured_mvc = kwargs.get("measured_mvc", None)
        if measured_mvc is not None:
            self.adapt_parameters(measured_mvc)

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
        x_ref = self.initial_pose + self.ref_vel * t

        # Safety: reset if velocity explodes
        if np.any(np.abs(self.xd) > 1000.0):
            self._log("⚠️ Warning: xd overflow — resetting to zero.")
            self.xd[:] = 0.0

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
        self._log(
            f"[AC] t={t:.2f} | x_ref: {x_ref[:3]} | force_z={self.force[2]:.2f} | ||tau||={np.linalg.norm(tau):.2f}"
        )
        return tau, self.force

    def adapt_parameters(self, measured_mvc):
        error = self.target_mvc - measured_mvc

        self.M += self.adapt_gain_m * error
        self.B += self.adapt_gain_b * error
        self.K += self.adapt_gain_k * error

        try:
            self.M = np.clip(self.M, self.M_min, self.M_max)
            self.B = np.clip(self.B, self.B_min, self.B_max)
            self.K = np.clip(self.K, self.K_min, self.K_max)
        except AttributeError:
            self._log("⚠️ Warning: Clipping parameters not set — skipping clipping.")
            return

        self._log(
            f"[AC] Adapted M={self.M:.2f}, B={self.B:.2f}, K={self.K:.2f} based on error={error:.2f}"
        )
