import mujoco
import mujoco.viewer
import mujoco.glfw
import numpy as np
import time
import matplotlib.pyplot as plt
from gui import ForceControlGUI



from controller import ParallelForceMotionController

z_force = 0.5

# Step 2: Load model and controller
model = mujoco.MjModel.from_xml_path("mujoco_menagerie/franka_fr3/fr3.xml")
data = mujoco.MjData(model)
controller = ParallelForceMotionController(model, data, site_name="attachment_site")
gui = ForceControlGUI()

# Step 3: Set goal pose - straight line down at x = 0.5 m from base
mujoco.mj_forward(model, data)
x_curr = data.site_xpos[controller.site_id].copy()
x_goal = x_curr.copy()
x_goal[0] = 0.5     # Set x to 0.5 m in front of base
x_goal[1] = 0.0     # Centered on y
x_goal[2] -= 1.0    # 10 cm downward


# Step 4: Setup data logging
force_log, vel_log, time_log = [], [], []

# Step 5: Launch viewer with key callback (optional)
start_time = time.time()
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        z_force = gui.get_force()
        controller.set_force_z(z_force)

        tau, F = controller.compute_torques(x_goal)
        data.ctrl[:] = tau

        mujoco.mj_step(model, data)
        viewer.sync()

        t = time.time() - start_time
        gui.update_plot(t, F[2], data.qvel)


# Step 6: Plot after simulation ends
plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1)
plt.plot(time_log, force_log, label="Z Force [N]")
plt.ylabel("Z Force (N)")
plt.grid()

