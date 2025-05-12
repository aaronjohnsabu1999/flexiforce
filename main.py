# main.py
import time
import os
import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
from threading import Thread
import tkinter as tk

from gui import ForceControlGUI
from controller import ParallelForceMotionController, AdmittanceController


def run_simulation(gui, model, data, controller, x_goal, dt, log):
    def simulate_loop():
        start_time = time.time()
        while True:
            force = gui.get_force()
            controller.set_force(force)

            M_new, B_new, K_new = gui.get_admittance_params()
            controller.M = max(0.1, min(M_new, 10.0))
            controller.B = max(0.0, min(B_new, 200.0))
            controller.K = max(0.0, min(K_new, 200.0))

            tau, F = controller.compute_torques(x_goal=x_goal, dt=dt)
            data.ctrl[:] = tau
            mujoco.mj_step(model, data)

            t = time.time() - start_time
            gui.update_plot(t, F[2], data.qvel)

            if t > 60.0:
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


if __name__ == "__main__":
    # Step 1: Initialize GUI
    gui = ForceControlGUI()
    gui.set_window(
        title="Force Control GUI", size=(800, 600), pos=(100, 100), color=(0.1, 0.1, 0.1)
    )

    # Step 2: Load model and controller
    model = mujoco.MjModel.from_xml_path("mujoco_menagerie/franka_fr3/fr3.xml")
    data = mujoco.MjData(model)

    controller = ParallelForceMotionController(model, data, site_name="attachment_site")

    # Step 3: Set goal pose
    mujoco.mj_forward(model, data)
    x_curr = data.site_xpos[controller.site_id].copy()
    x_goal = x_curr.copy()
    x_goal[0] = 0.5
    x_goal[1] = 0.0
    x_goal[2] -= 1.0
    dt = 0.01

    # Step 4: Prepare shared log for results
    log = {"time": [], "force": [], "vel": []}

    # Step 5: Launch simulation in background thread
    sim_thread = Thread(target=run_simulation, args=(gui, model, data, controller, x_goal, dt, log))
    sim_thread.start()

    # Step 6: Start GUI mainloop in main thread
    try:
        tk.mainloop()
    finally:
        # gui.stop()
        sim_thread.join()

    # Step 7: Final results plot
    plt.figure(figsize=(10, 5), num="Simulation Results")
    plt.subplot(2, 1, 1)
    plt.plot(log["time"], log["force"], label="Z Force [N]")
    plt.ylabel("Z Force (N)")
    plt.grid()
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(log["time"], [v[0] for v in log["vel"]], label="Joint 1 Velocity")
    plt.ylabel("Joint Velocity (rad/s)")
    plt.xlabel("Time (s)")
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.show()
