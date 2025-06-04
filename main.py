import os
import time
import yaml
import mujoco
import mujoco.viewer
import numpy as np
from controller import AdmittanceController
from trajectories import get_trajectory_functions

def load_config():
    try:
        with open("config.yaml", "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print("⚠️ config.yaml not found. Exiting.")
        exit()

def run():
    config = load_config()
    model_path = config["simulation"]["model_path"]

    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return

    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    controller = AdmittanceController(
        model, data,
        site_name=config["simulation"]["site_name"],
        M=np.diag(config["admittance_controller"]["M"]),
        B=np.diag(config["admittance_controller"]["B"]),
        K=np.diag(config["admittance_controller"]["K"]),
        Kp=config["admittance_controller"]["Kp"],
        Kd=config["admittance_controller"]["Kd"],
        use_contact_forces=config["admittance_controller"]["use_contact_forces"],
        contact_geom_ids=config["admittance_controller"]["contact_geom_ids"],
    )

    # Home configuration
    data.qpos[:] = np.array([0.0, -0.3, 0.0, -1.57, 3.14, 1.57, 0.0])
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    # Set desired force
    controller.set_desired_force(config["forces"]["desired_force"])

    # Set trajectory
    traj_func, vel_func = get_trajectory_functions(config)
    controller.set_reference_trajectory(traj_func, vel_func)

    dt = config["simulation"]["dt"]
    duration = config["simulation"]["duration"]
    log = {"time": [], "force": [], "vel": [], "position":[]}

    with mujoco.viewer.launch_passive(model, data) as viewer:
        t0 = time.time()
        while viewer.is_running() and time.time() - t0 < duration:
            t = time.time() - t0

            tau, measured_force = controller.compute_torques(dt, t)
            data.ctrl[:] = tau
            mujoco.mj_step(model, data)
            viewer.sync()

            log["time"].append(t)
            log["force"].append(measured_force[2])
            log["vel"].append(data.qvel.copy())
            log["position"].append(controller.x_adm[:3].copy())
            
            time.sleep(0.001)

    log["time"] = np.array(log["time"])
    log["force"] = np.array(log["force"])
    log["vel"] = np.vstack(log["vel"]) if log["vel"] else np.array([])
    log["position"] = np.array(log["position"])
    controller.plot_results(log)

if __name__ == "__main__":
    run()

