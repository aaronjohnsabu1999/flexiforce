import mujoco
import mujoco.viewer
import mujoco.glfw
import numpy as np
import time
import matplotlib.pyplot as plt
from gui import ForceControlGUI

from controller import ParallelForceMotionController, AdmittanceController

# Step 1: Initialize GUI
gui = ForceControlGUI()
gui.set_window(
    title="Force Control GUI", size=(800, 600), pos=(100, 100), color=(0.1, 0.1, 0.1)
)

# Step 2: Load model and controller
model = mujoco.MjModel.from_xml_path("mujoco_menagerie/franka_fr3/fr3.xml")
data = mujoco.MjData(model)

# Initialize the controller
USE_ADMITTANCE = False
if USE_ADMITTANCE:
    M, B, K = 1.0, 50.0, 0.0
    Kp, Kd = 2000.0, 50.0
    controller = AdmittanceController(
        model, data, site_name="attachment_site", M=M, B=B, K=K, Kp=Kp, Kd=Kd
    )
else:
    # Use ParallelForceMotionController for force control
    controller = ParallelForceMotionController(model, data, site_name="attachment_site")

# Step 3: Set goal pose - straight line down at x = 0.5 m from base
mujoco.mj_forward(model, data)
x_curr = data.site_xpos[controller.site_id].copy()
x_goal = x_curr.copy()
x_goal[0] = 0.5  # Set x to 0.5 m in front of base
x_goal[1] = 0.0  # Centered on y
x_goal[2] -= 1.0  # 10 cm downward

# Step 4: Setup data logging
force_log, vel_log, time_log = [], [], []

# Step 5: Launch viewer with key callback (optional)
start_time = time.time()
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        force = gui.get_force()
        controller.set_force(force)

        tau, F = controller.compute_torques(x_goal)
        data.ctrl[:] = tau

        mujoco.mj_step(model, data)
        viewer.sync()

        t = time.time() - start_time
        gui.update_plot(t, F[2], data.qvel)

        # Optional: log data
        time_log.append(t)
        force_log.append(F[2])
        vel_log.append(data.qvel.copy())

# Step 6: Plot after simulation ends
plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1)
plt.plot(time_log, force_log, label="Z Force [N]")
plt.ylabel("Z Force (N)")
plt.grid()
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(time_log, [v[0] for v in vel_log], label="Joint 1 Velocity")
plt.ylabel("Joint Velocity (rad/s)")
plt.xlabel("Time (s)")
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()
