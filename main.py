import os
import time
import yaml
import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
from controller import AdmittanceController
from trajectories import get_trajectory_functions

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def plot_results(log):
    plt.figure(figsize=(15, 8))

    plt.subplot(2, 1, 1)
    for i, label in enumerate(['X', 'Y', 'Z']):
        plt.plot(log["time"], log["x_ref"][:, i], linestyle='--', label=f"{label} Reference")
        plt.plot(log["time"], log["measured_position"][:, i], label=f"{label} Measured")
    plt.ylabel("Position (m)")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(log["time"], log["applied_force"][:, 2], label="Applied External Z Force")
    plt.ylabel("Force (N)")
    plt.xlabel("Time (s)")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

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
        Kd=config["admittance_controller"]["Kd"]
    )

    data.qpos[:] = np.array(config["simulation"]["initial_qpos"])
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    controller.set_external_force(config["forces"]["external_force"],0)
    traj_func, vel_func = get_trajectory_functions(config)
    controller.set_trajectory(traj_func, vel_func)

    dt = config["simulation"]["dt"]
    duration = config["simulation"]["duration"]
    log = {"time": [], "force": [], "vel": [], "position":[], "applied_force": [], "measured_position": [], "x_ref": []}

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, config["simulation"]["site_name"])
    body_id = model.site_bodyid[site_id]

    with mujoco.viewer.launch_passive(model, data) as viewer:
        t0 = time.time()
        while viewer.is_running() and time.time() - t0 < duration:
            t = time.time() - t0
            controller.set_external_force(config["forces"]["external_force"],t)
            tau = controller.compute_torques(t, dt)
            data.ctrl[:] = tau
            mujoco.mj_step(model, data)
            viewer.sync()

            log["time"].append(t)
            log["force"].append(controller.external_force.copy())
            log["vel"].append(data.qvel.copy())
            log["position"].append(controller.x[:3].copy())
            log["applied_force"].append(controller.external_force.copy())

            _, _, pose_meas = controller.get_current_pose()
            log["measured_position"].append(pose_meas[:3].copy())
            x_ref, _ = controller.get_reference_at_time(t)
            log["x_ref"].append(x_ref[:3].copy())
            time.sleep(0.001)

    log["time"] = np.array(log["time"])
    log["force"] = np.vstack(log["force"])
    log["vel"] = np.vstack(log["vel"]) if log["vel"] else np.array([])
    log["position"] = np.array(log["position"])
    log["applied_force"] = np.vstack(log["applied_force"])
    log["measured_position"] = np.array(log["measured_position"])
    log["x_ref"] = np.array(log["x_ref"])

    plot_results(log)

if __name__ == "__main__":
    run()
