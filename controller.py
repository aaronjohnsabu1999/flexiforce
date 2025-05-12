# controller.py
import numpy as np
import mujoco


class Controller:
    def __init__(self, model, data, site_name, verbose=False):
        self.model = model
        self.data = data
        self.site_id = model.site(site_name).id
        self.dt = 0.01  # Time step for simulation
        self.verbose = verbose
        self.force = np.zeros(3)  # External force vector

    def set_force(self, f_ext):
        """Set the external force vector."""
        self.force = np.array(f_ext)

    def compute_torques(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement compute_torques")


class ParallelForceMotionController(Controller):
    def __init__(self, model, data, site_name, **kwargs):
        super().__init__(model, data, site_name, **kwargs)
        self.Kp = kwargs.get("Kp", 250.0)
        self.Kd = kwargs.get("Kd", 5.0)

    def compute_torques(self, *args, **kwargs):
        x_goal = kwargs.get("x_goal", None)
        dt = kwargs.get("dt", self.dt)

        mujoco.mj_forward(self.model, self.data)
        x_curr = self.data.site_xpos[self.site_id]
        dx = x_goal[:2] - x_curr[:2]  # xy control only

        J_pos = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, J_pos, None, self.site_id)

        F = np.zeros(3)
        F[:2] = self.Kp * dx
        F[2] = self.force[2]  # Only apply z-component from force input

        tau = J_pos.T @ F - self.Kd * self.data.qvel
        if self.verbose:
            print(
                f"[PFC] dx: {dx}, force: {self.force}, tau norm: {np.linalg.norm(tau):.2f}"
            )
        return tau, F


class AdmittanceController(Controller):
    def __init__(self, model, data, site_name, *args, **kwargs):
        verbose = kwargs.get("verbose", False)
        super().__init__(model, data, site_name, verbose=verbose)
        # Admittance parameters
        self.M = kwargs.get("M", 1.0)  # Mass
        self.B = kwargs.get("B", 50.0)  # Damping
        self.K = kwargs.get("K", 0.0)  # Stiffness

        # Velocity control gains
        self.Kp = kwargs.get("Kp", 100.0)  # Proportional gain
        self.Kd = kwargs.get("Kd", 0.0)  # Derivative gain

        # State
        self.xd = np.zeros(3)
        self.x = None
        self.force = np.zeros(3)

    def compute_torques(self, *args, **kwargs):
        x_goal = kwargs.get("x_goal", None)
        dt = kwargs.get("dt", self.dt)

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
        if self.verbose:
            print(
                f"[AC] x: {self.x}, xd: {self.xd}, force: {self.force}, tau norm: {np.linalg.norm(tau):.2f}"
            )
        return tau, self.force
