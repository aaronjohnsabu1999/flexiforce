# simulation.py
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
from threading import Thread
import logging


class Simulation:
    def __init__(
        self,
        config,
        gui,
        controller,
        model,
        data,
        x_goal,
        dt,
        verbose=False,
        opensim_trajectory=None,
    ):
        self.config = config
        self.gui = gui
        self.controller = controller
        self.model = model
        self.data = data
        self.x_goal = x_goal
        self.dt = dt
        self.verbose = verbose
        self.log = {"time": np.array([]), "force": np.array([]), "vel": np.array([])}
        self._stop = False
        self.opensim_trajectory = opensim_trajectory

    def stop(self):
        self._stop = True

    def step(self):
        self.desired_force = np.zeros(6)
        start_time = time.time()
        while not self._stop:
            target_mvc = self.gui.get_target_mvc() if self.gui else 50.0
            max_force = (
                self.gui.get_max_force()
                if self.gui
                else self.config["simulation"]["max_force"]
            )

            t = time.time() - start_time
            if self.opensim_trajectory:
                self.desired_force[2] = self.opensim_trajectory.get_force_at(t)
            else:
                self.desired_force[2] = -(target_mvc / 100.0) * max_force
            self.controller.set_force(self.desired_force, target_mvc=target_mvc)

            if self.gui and self.gui.is_admittance_controller():
                M_new, B_new, K_new = self.gui.get_admittance_params()
                self.controller.M = max(
                    self.config["admittance_controller"]["M_min"],
                    min(M_new, self.config["admittance_controller"]["M_max"]),
                )
                self.controller.B = max(
                    self.config["admittance_controller"]["B_min"],
                    min(B_new, self.config["admittance_controller"]["B_max"]),
                )
                self.controller.K = max(
                    self.config["admittance_controller"]["K_min"],
                    min(K_new, self.config["admittance_controller"]["K_max"]),
                )

            actual_torque, actual_force = self.controller.compute_torques(
                dt=self.dt, x_goal=self.x_goal, time_now=t
            )
            self.data.ctrl[:] = actual_torque
            mujoco.mj_step(self.model, self.data)

            if self.gui:
                self.gui.update_plot(t, actual_force[2], self.data.qvel)

            if self.verbose:
                if self.gui:
                    self.gui.log_slider_changes()
                logging.info(
                    f"t={t:.2f} | MVC={target_mvc:.3f}% | force_z={actual_force[2]:.3f} | vel_max={np.max(np.abs(self.data.qvel)):.3f}"
                )
            if t > self.config["simulation"]["duration"]:
                logging.info("Simulation duration reached, stopping.")
                break

            self.log["time"] = np.append(self.log["time"], t)
            self.log["force"] = np.append(self.log["force"], actual_force[2])
            if self.log["vel"].size == 0:
                self.log["vel"] = np.array([self.data.qvel.copy()])
            else:
                self.log["vel"] = np.vstack([self.log["vel"], self.data.qvel.copy()])

            yield

    def run(self):
        # Viewer/Headless selection
        if "DISPLAY" in os.environ and os.environ["DISPLAY"]:
            try:
                with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                    for _ in self.step():
                        viewer.sync()
            except Exception as e:
                logging.error(f"MuJoCo viewer failed: {e}\nRunning headless instead.")
                logging.error("Common causes:")
                logging.error("  - X server not running on Windows (e.g., VcXsrv)")
                logging.error("  - DISPLAY not set in WSL (try: export DISPLAY=:0)")
                logging.error("  - OpenGL drivers missing or blocked")
                for _ in self.step():
                    pass
        else:
            logging.warning("No DISPLAY detected — running headless.")
            for _ in self.step():
                pass

    def plot_results(self):
        plt.figure(figsize=(10, 5), num="Simulation Results")
        plt.subplot(2, 1, 1)
        # plt.plot(log["time"], log["force"], label="Z Force [N]")
        plt.plot(self.log["time"], self.log["force"], label="Z Force [N]")
        plt.ylabel("Z Force (N)")
        plt.grid()
        plt.legend()

        plt.subplot(2, 1, 2)
        for i in range(self.data.qvel.shape[0]):
            plt.plot(
                self.log["time"],
                self.log["vel"][:, i],
                label=f"Joint {i+1} Velocity",
            )
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.show()
