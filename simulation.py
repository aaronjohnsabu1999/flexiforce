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
    def __init__(self, config, gui, controller, model, data, x_goal, dt, verbose=False):
        self.config = config
        self.gui = gui
        self.controller = controller
        self.model = model
        self.data = data
        self.x_goal = x_goal
        self.dt = dt
        self.verbose = verbose
        self.log = {"time": [], "force": [], "vel": []}
        self._stop = False

    def stop(self):
        self._stop = True

    def simulate_loop(self):
        start_time = time.time()
        while not self._stop:
            force = self.gui.get_force()
            target_mvc = self.gui.get_target_mvc()
            max_force = self.config["simulation"]["max_force"]
            desired_force_z = -(target_mvc / 100.0) * max_force
            force[2] = desired_force_z
            self.controller.set_force(force, target_mvc=target_mvc)

            M_new, B_new, K_new = self.gui.get_admittance_params()
            self.controller.M = max(0.1, min(M_new, 10.0))
            self.controller.B = max(0.0, min(B_new, 200.0))
            self.controller.K = max(0.0, min(K_new, 200.0))

            tau, F = self.controller.compute_torques(x_goal=self.x_goal, dt=self.dt)
            self.data.ctrl[:] = tau
            mujoco.mj_step(self.model, self.data)

            t = time.time() - start_time
            self.gui.update_plot(t, F[2], self.data.qvel)

            if self.verbose:
                logging.info(
                    f"t={t:.2f} | MVC={target_mvc}% | force_z={F[2]:.2f} | vel_max={np.max(np.abs(self.data.qvel)):.2f}"
                )
            if t > 60.0:
                break

            self.log["time"].append(t)
            self.log["force"].append(F[2])
            self.log["vel"].append(self.data.qvel.copy())

            yield

    def run(self):
        # Viewer/Headless selection
        if "DISPLAY" in os.environ and os.environ["DISPLAY"]:
            try:
                with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                    for _ in self.simulate_loop():
                        viewer.sync()
            except Exception as e:
                logging.error(f"MuJoCo viewer failed: {e}\nRunning headless instead.")
                for _ in self.simulate_loop():
                    pass
        else:
            logging.warning("No DISPLAY detected — running headless.")
            for _ in self.simulate_loop():
                pass

    def plot_results(self):
        plt.figure(figsize=(10, 5), num="Simulation Results")
        plt.subplot(2, 1, 1)
        plt.plot(self.log["time"], self.log["force"], label="Z Force [N]")
        plt.ylabel("Z Force (N)")
        plt.grid()
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(self.log["time"], [v[0] for v in self.log["vel"]], label="Joint 1 Velocity")
        plt.ylabel("Joint Velocity (rad/s)")
        plt.xlabel("Time (s)")
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.show()
