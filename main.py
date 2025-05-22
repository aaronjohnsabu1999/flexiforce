# main.py
import os
import time
import yaml
import mujoco
import mujoco.viewer
import argparse
import numpy as np
import matplotlib.pyplot as plt
from threading import Thread
import tkinter as tk
from scipy.spatial.transform import Rotation as R

from controller import AdmittanceController


with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


def run_simulation(model, data, controller, dt, log, force_input, verbose=False):
    def simulate_loop():
        start_time = time.time()
        while True:

            controller.set_force(force_input)

            t = time.time() - start_time
            tau, F = controller.compute_torques(dt=dt, time=t)
            data.ctrl[:] = tau
            mujoco.mj_step(model, data)


            if verbose:
                print(
                    f"t={t:.2f} | force_z={F[2]:.2f} | vel_max={np.max(np.abs(data.qvel)):.2f}"
                )
            if t > 5.0:
                break

            log["time"].append(t)
            log["force"].append(F[2])
            log["vel"].append(data.qvel.copy())

            yield  # sync point for viewer

    # Decide viewer or headless
    if "DISPLAY" in os.environ and os.environ["DISPLAY"]:
        try:
            with mujoco.viewer.launch_passive(model, data) as viewer:
                for _ in simulate_loop():
                    viewer.sync()
        except Exception as e:
            print("\n❌ MuJoCo viewer failed to launch.")
            print("→ Common causes:")
            print("  - X server not running on Windows (e.g., VcXsrv)")
            print("  - DISPLAY not set in WSL (try: export DISPLAY=:0)")
            print("  - OpenGL drivers missing or blocked")
            print(f"Error: {e}\n→ Running headless instead.")
            for _ in simulate_loop():
                pass
    else:
        print("⚠️ No DISPLAY detected — running headless.")
        for _ in simulate_loop():
            pass


parser = argparse.ArgumentParser(description="Run FlexiForce simulation.")
parser.add_argument("--verbose", action="store_true", help="Enable verbose output.")
args = parser.parse_args()
VERBOSE = args.verbose

if __name__ == "__main__":
    default_force_z = -config["simulation"]["max_force"] * 0.5  # simulate 50% MVC TODO replace with the gui and Opensim values
    default_force = np.zeros(6)
    default_force[2] = default_force_z  # Z-force only

    # Step 2: Load model and controller
    model = mujoco.MjModel.from_xml_path(config["simulation"]["model_path"])
    data = mujoco.MjData(model)

    USE_ADMITTANCE = True
    if USE_ADMITTANCE:
        controller = AdmittanceController(
            model,
            data,
            site_name="attachment_site",
            **config["admittance_controller"],
            verbose=VERBOSE,
        )

    # Step 3: Set goal pose
    # Set home configuration manually (in radians)
    home_q = np.array([0.0, -0.3, 0.0, -1.57, 0.0, 1.57, 0.0])
    data.qpos[:] = home_q
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    x_curr = data.site_xpos[controller.site_id].copy()
    x_goal = np.array(config["simulation"]["x_goal"])
    dt = config["simulation"]["dt"]

    # Get current orientation in rotation matrix
    R_curr = data.site_xmat[controller.site_id].reshape((3, 3))
    rotvec_curr = R.from_matrix(R_curr).as_rotvec()

    # Step 4: Prepare shared log for results
    log = {
    "time": [],
    "force": [],       # scalar z-force
    "vel": []}


    # Step 5: Launch simulation in background thread
    run_simulation(model, data, controller, dt, log, default_force, VERBOSE)

    # Convert to NumPy arrays after the run
    log["time"] = np.array(log["time"])       # shape (T,)
    log["force"] = np.array(log["force"])     # shape (T,)
    log["vel"] = np.vstack(log["vel"])        # shape (T, 7)

    # Step 7: Final results plot
    plt.figure(figsize=(10, 5), num="Simulation Results")
    plt.subplot(2, 1, 1)
    plt.plot(log["time"], log["force"], label="Z Force [N]")
    plt.ylabel("Z Force (N)")
    plt.grid()
    plt.legend()

    plt.subplot(2, 1, 2)
    for j in range(7):
        plt.plot(log["time"], log["vel"][:, j], label=f'Joint {j+1}')
    plt.ylabel("Joint Velocity (rad/s)")
    plt.xlabel("Time (s)")
    plt.grid()
    plt.legend(loc="upper right")

    plt.tight_layout()
    plt.show()
