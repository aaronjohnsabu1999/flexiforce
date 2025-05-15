# controller.py
import numpy as np
import mujoco


class Controller:
    def __init__(self, model, data, site_name, verbose=False):
        self.type = None
        self.model = model
        self.data = data
        self.site_id = model.site(site_name).id
        self.dt = 0.01
        self.verbose = verbose
        self.force = np.zeros(3)
        self.target_mvc = 0.0

    def set_force(self, f_ext, target_mvc=None):
        self.force = np.array(f_ext)
        if target_mvc is not None:
            self.target_mvc = target_mvc

    def compute_torques(self, x_goal=None, dt=None, **kwargs):
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

    def compute_torques(self, x_goal=None, dt=None, **kwargs):
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
        self.M = kwargs.get("M", 1.0)  # Mass
        self.B = kwargs.get("B", 50.0)  # Damping
        self.K = kwargs.get("K", 0.0)  # Stiffness

        # Velocity control gains
        self.Kp = kwargs.get("Kp", 100.0)  # Proportional gain
        self.Kd = kwargs.get("Kd", 0.0)  # Derivative gain

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
        self.xd = np.zeros(3)
        self.x = None
        self.force = np.zeros(3)

    def compute_torques(self, x_goal=None, dt=None, **kwargs):
        if dt is not None:
            self.dt = dt

        measured_mvc = kwargs.get("measured_mvc", None)
        if measured_mvc is not None:
            self.adapt_parameters(measured_mvc)

        mujoco.mj_forward(self.model, self.data)
        x_now = self.data.site_xpos[self.site_id]

        if self.x is None:
            self.x = x_now.copy()

        # Admittance dynamics: M ẍ + B ẋ + K x = F_ext
        acc = (self.force - self.B * self.xd - self.K * (self.x - x_now)) / self.M
        self.xd += acc * self.dt
        self.x += self.xd * self.dt

        # Safety: reset if velocity explodes
        if np.any(np.abs(self.xd) > 1000.0):
            self._log("⚠️ Warning: xd overflow — resetting to zero.")
            self.xd[:] = 0.0

        # Compute Jacobian
        J_pos = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, J_pos, None, self.site_id)

        # Map Cartesian velocity to joint velocity
        qvel_desired = np.linalg.pinv(J_pos) @ self.xd

        # Velocity-level control in joint space
        vel_error = qvel_desired - self.data.qvel
        tau = self.Kp * vel_error - self.Kd * self.data.qvel
        self._log(
            f"[AC] x: {self.x}, xd: {self.xd}, force: {self.force}, tau norm: {np.linalg.norm(tau):.2f}"
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
