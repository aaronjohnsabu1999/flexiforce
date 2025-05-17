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

from gui import ForceControlGUI
from opensim_force_trajectory import OpenSimForceTrajectory
from controller import ParallelForceMotionController, AdmittanceController


with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
enable_viewer = (
    config["simulation"].get("enable_viewer", True)
    and "DISPLAY" in os.environ
    and os.environ["DISPLAY"]
)
opensim_traj = OpenSimForceTrajectory(
    "./OpenSIM_utils/Arm26/OutputReference/ComputedMuscleControl/Results/arm26_Actuation_force.sto"
)


parser = argparse.ArgumentParser(description="Run FlexiForce simulation.")
parser.add_argument("--verbose", action="store_true", help="Enable verbose output.")
args = parser.parse_args()
VERBOSE = args.verbose


def update_controller_params(controller, gui):
    if controller.type == "AC":
        M_new, B_new, K_new = gui.get_admittance_params()
        controller.M = np.clip(M_new, controller.M_min, controller.M_max)
        controller.B = np.clip(B_new, controller.B_min, controller.B_max)
        controller.K = np.clip(K_new, controller.K_min, controller.K_max)


def run_simulation(gui, model, data, controller, x_goal, dt, log, verbose=VERBOSE):
    def simulate_loop():
        start_time = time.time()
        while True:
            t = time.time() - start_time

            # Get user input
            force = gui.get_force()
            user_force_z = force[2]

            # Get max force from user-defined MVC
            target_mvc = gui.get_target_mvc()
            max_force = gui.get_max_force()
            scaled_user_force_z = (target_mvc / 100.0) * max_force

            # Get ideal force from OpenSim
            opensim_force_z = opensim_traj.get_force_at(t)

            # Cap the user-applied force based on what is biomechanically safe
            force[2] = min(scaled_user_force_z, opensim_force_z)
            controller.set_force(force, target_mvc=target_mvc)
            update_controller_params(controller, gui)

            # Simulate one step
            tau, F = controller.compute_torques(x_goal=x_goal, dt=dt)
            data.ctrl[:] = tau
            mujoco.mj_step(model, data)

            # Update GUI and logs
            gui.update_plot(t, F[2], data.qvel)

            if verbose:
                print(
                    f"t={t:.2f} | user={scaled_user_force_z:.2f} N | safe={opensim_force_z:.2f} N | applied={force[2]:.2f} N | vel_max={np.max(np.abs(data.qvel)):.2f}"
                )

            if t > 120.0:
                break

            log["time"].append(t)
            log["vel"].append(data.qvel.copy())
            log["force"].append(F[2])
            log["force_user"].append(scaled_user_force_z)
            log["force_safe"].append(opensim_force_z)

            yield  # sync point for viewer

    # Decide viewer or headless
    if enable_viewer:
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


if __name__ == "__main__":
    # Step 1: Initialize GUI
    gui = ForceControlGUI(
        verbose=VERBOSE,
        init_force=np.array(config["gui"]["init_force"]),
        sliders=config["gui"]["sliders"],
        controller=config["simulation"]["controller"],
        enable_viewer=enable_viewer,
    )
    gui.set_window(**config["gui"]["window"])

    # Step 2: Load model and controller
    model = mujoco.MjModel.from_xml_path(config["simulation"]["model_path"])
    data = mujoco.MjData(model)

    if config["simulation"].get("controller") == "AC":
        controller = AdmittanceController(
            model,
            data,
            site_name="attachment_site",
            **config["admittance_controller"],
            verbose=VERBOSE,
        )
    elif config["simulation"].get("controller") == "PFC":
        controller = ParallelForceMotionController(
            model,
            data,
            site_name="attachment_site",
            **config["parallel_controller"],
            verbose=VERBOSE,
        )
        controller.set_force(config["parallel_controller"]["force"])
    else:
        raise ValueError(
            "Invalid controller type. Choose 'AC' for Admittance or 'PFC' for Parallel Force Control."
        )

    # Step 3: Set goal pose
    mujoco.mj_forward(model, data)
    x_curr = data.site_xpos[controller.site_id].copy()
    x_offset = np.array(config["simulation"]["x_goal_offset"])
    x_goal = x_curr + x_offset
    dt = config["simulation"]["dt"]

    # Step 4: Prepare shared log for results
    log = {"time": [], "vel": [], "force": [], "force_user": [], "force_safe": []}

    # Step 5: Launch simulation in background thread
    sim_thread = Thread(
        target=run_simulation,
        args=(gui, model, data, controller, x_goal, dt, log, VERBOSE),
    )
    sim_thread.start()

    # Step 6: Start GUI mainloop in main thread
    try:
        tk.mainloop()
    finally:
        # gui.stop()
        sim_thread.join()

    # Step 7: Final results plot
    plt.figure(figsize=(10, 5), num="Simulation Results")

    # Force plot
    plt.subplot(2, 1, 1)
    plt.plot(
        log["time"],
        log["force"],
        label="Z Force [N] (Applied)",
        linestyle="-",
        color="black",
    )
    plt.plot(
        log["time"],
        log["force_user"],
        label="User Force [N]",
        linestyle="--",
        color="tab:blue",
    )
    plt.plot(
        log["time"],
        log["force_safe"],
        label="Safe Force [N]",
        linestyle=":",
        color="tab:green",
    )
    plt.ylabel("Z Force (N)")
    plt.grid(True)
    plt.legend()

    # Velocity plot
    plt.subplot(2, 1, 2)
    plt.plot(
        log["time"],
        [v[0] for v in log["vel"]],
        label="Joint 1 Velocity",
        linestyle="-.",
        color="tab:red",
    )
    plt.ylabel("Joint Velocity (rad/s)")
    plt.xlabel("Time (s)")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()
