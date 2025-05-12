import mujoco
import mujoco.viewer
import mujoco.glfw
import numpy as np
import time
import matplotlib.pyplot as plt
from gui import ForceControlGUI

from controller import ParallelForceMotionController, AdmittanceController

z_force = 0.5

# Step 2: Load model and controller
model = mujoco.MjModel.from_xml_path("mujoco_menagerie/franka_fr3/fr3.xml")
data = mujoco.MjData(model)
M,B,K=1.0,50.0,0.0
Kp,Kd=2000.0,50.0
# controller = ParallelForceMotionController(model, data, site_name="attachment_site")
controller = AdmittanceController(
    model, data, site_name="attachment_site", M=M, B=B, K=K, Kp=Kp, Kd=Kd
)
gui = ForceControlGUI()

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
while True:
    z_force = gui.get_force() if gui else -5.0
    controller.set_external_force([0.0, 0.0, z_force])

    M_new, B_new, K_new = gui.get_admittance_params() if gui else (1.0, 50.0, 0.0)
    controller.M = max(0.1, min(M_new, 10.0))
    controller.B = max(0.0, min(B_new, 200.0))
    controller.K = max(0.0, min(K_new, 200.0))

    tau, F = controller.compute_torques(1 / 60.0)
    data.ctrl[:] = tau
    mujoco.mj_step(model, data)

    t = time.time() - start_time
    if gui:
        gui.update_plot(t, F[2], data.qvel)

    if t > 5.0:
        break

# Step 6: Plot after simulation ends
plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1)
plt.plot(time_log, force_log, label="Z Force [N]")
plt.ylabel("Z Force (N)")
plt.grid()
