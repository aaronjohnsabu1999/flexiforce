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
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # GUI setup
    gui = ForceControlGUI(
        verbose=args.verbose,
        init_force=np.array(config["gui"]["init_force"]),
        sliders=config["gui"]["sliders"],
        controller=config["simulation"]["controller"],
        enable_viewer=enable_viewer,
    )
    gui.set_window(**config["gui"]["window"])

    # Model/controller
    model = mujoco.MjModel.from_xml_path(config["simulation"]["model_path"])
    data = mujoco.MjData(model)

    if config["simulation"].get("controller") == "AC":
        controller = AdmittanceController(
            model,
            data,
            site_name="attachment_site",
            **config["admittance_controller"],
            verbose=args.verbose,
        )
    elif config["simulation"].get("controller") == "PFC":
        controller = ParallelForceMotionController(
            model,
            data,
            site_name="attachment_site",
            **config["parallel_controller"],
            verbose=args.verbose,
        )
        controller.set_force(config["parallel_controller"]["force"])
    else:
        raise ValueError(
            "Invalid controller type. Choose 'AC' for Admittance or 'PFC' for Parallel Force Control."
        )

    mujoco.mj_forward(model, data)
    x_curr = data.site_xpos[controller.site_id].copy()
    x_goal = x_curr + np.array(config["simulation"]["x_goal_offset"])
    dt = config["simulation"]["dt"]

    sim = Simulation(
        config, gui, controller, model, data, x_goal, dt, verbose=args.verbose
    )
    sim_thread = Thread(target=sim.run)
    sim_thread.start()

    try:
        tk.mainloop()
    finally:
        sim.stop()
        sim_thread.join()
        sim.plot_results()
