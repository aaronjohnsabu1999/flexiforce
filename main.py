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

    wall_geom_name = "virtual_wall"
    wall_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, wall_geom_name)

    controller = AdmittanceController(
        model, data,
        site_name=config["simulation"]["site_name"],
        M=np.diag(config["admittance_controller"]["M"]),
        B=np.diag(config["admittance_controller"]["B"]),
        K=np.diag(config["admittance_controller"]["K"]),
        Kp=config["admittance_controller"]["Kp"],
        Kd=config["admittance_controller"]["Kd"],
        use_contact_forces=False,
        contact_geom_ids=[],
    )

    data.qpos[:] = np.array([0.002, -1.26,  0.01, -2.31, -0.05,  2.06,  0.83])
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    controller.set_desired_force(config["forces"]["desired_force"])

    traj_func, vel_func = get_trajectory_functions(config)
    controller.set_reference_trajectory(traj_func, vel_func)

    dt = config["simulation"]["dt"]
    duration = config["simulation"]["duration"]
    log = {"time": [], "force": [], "vel": [], "position":[], "measured_position": [], "x_ref": []}

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, config["simulation"]["site_name"])
    body_id = model.site_bodyid[site_id]

    with mujoco.viewer.launch_passive(model, data) as viewer:
        t0 = time.time()
        while viewer.is_running() and time.time() - t0 < duration:
            t = time.time() - t0

            #data.xfrc_applied[body_id][:] = np.array([0, 0, -10, 0, 0, 0])

            tau, measured_force = controller.compute_torques(dt, t)

            if abs(t % 1.0) < dt:
                x_ref, _ = controller.get_reference_at_time(t)
                x_adm = controller.x_adm
                force_error = controller.force_desired - controller.force_measured
                tau_norm = np.linalg.norm(tau)

                print(f"\n[DEBUG @ t={t:.2f}s]")
                print(f"  x_ref       = {x_ref[:3]}")
                print(f"  x_adm       = {x_adm[:3]}")
                print(f"  force_error = {force_error[:3]}")
                print(f"  ||tau||     = {tau_norm:.4f}")

            data.ctrl[:] = tau
            mujoco.mj_step(model, data)
            viewer.sync()

            log["time"].append(t)
            log["force"].append(controller.force_desired[2])
            log["vel"].append(data.qvel.copy())
            log["position"].append(controller.x_adm[:3].copy())

            pos_meas, _ = controller.get_current_pose()
            log["measured_position"].append(pos_meas.copy())
            x_ref, _ = controller.get_reference_at_time(t)
            log["x_ref"].append(x_ref[:3].copy())

            time.sleep(0.001)

    log["time"] = np.array(log["time"])
    log["force"] = np.array(log["force"])
    log["vel"] = np.vstack(log["vel"]) if log["vel"] else np.array([])
    log["position"] = np.array(log["position"])
    log["measured_position"] = np.array(log["measured_position"])
    log["x_ref"] = np.array(log["x_ref"])
    controller.plot_results(log)

if __name__ == "__main__":
    run()
