# main.py
import argparse
import yaml
import mujoco
import numpy as np
import tkinter as tk
from threading import Thread
import logging

from gui import ForceControlGUI
from opensim_force_trajectory import OpenSimForceTrajectory
from controller import ParallelForceMotionController, AdmittanceController
from simulation import Simulation

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FlexiForce simulation.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output.")
    parser.add_argument(
        "--gui", action="store_true", help="Run with GUI (default: False)."
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    model = mujoco.MjModel.from_xml_path(config["simulation"]["model_path"])
    data = mujoco.MjData(model)

    if config["simulation"].get("controller") == "AC":
        controller = AdmittanceController(
            model,
            data,
            site_name="attachment_site",
            **config["admittance_controller"],
            verbose=args.verbose
        )
    elif config["simulation"].get("controller") == "PFC":
        controller = ParallelForceMotionController(
            model,
            data,
            site_name="attachment_site",
            **config["parallel_controller"],
            verbose=args.verbose
        )
        controller.set_force(config["parallel_controller"]["force"])
    else:
        raise ValueError("Invalid controller type. Choose 'AC' or 'PFC'.")

    data.qpos[:] = np.array([0.0, -0.3, 0.0, -1.57, 0.0, 1.57, 0.0])
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    x_curr = data.site_xpos[controller.site_id].copy()
    x_goal = np.array(config["simulation"]["x_goal"])
    dt = config["simulation"]["dt"]

    gui = None
    if args.gui:
        gui = ForceControlGUI(
            verbose=args.verbose,
            sliders=config["gui"]["sliders"],
            controller=controller,
            enable_viewer=True,
        )
        gui.set_window(**config["gui"]["window"])

    sim = Simulation(
        config, gui, controller, model, data, x_goal, dt, verbose=args.verbose
    )
    sim_thread = Thread(target=sim.run)
    sim_thread.start()

    try:
        if gui:
            tk.mainloop()
        else:
            sim_thread.join()  # headless mode, no GUI loop
    finally:
        sim.stop()
        sim_thread.join()
        sim.plot_results()
