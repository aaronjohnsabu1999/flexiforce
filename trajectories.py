import numpy as np

# Example time-variant trajectory functions
def quarter_circle_trajectory(t, center=[0.5, 0.0, 0.4], radius=0.2, frequency=1.0):
    theta = (3 * np.pi / 2) + (np.pi / 2) * (frequency * t % 1.0)
    x = center[0] + radius * np.cos(theta)
    y = center[1]
    z = center[2] + radius * np.sin(theta)
    return [x, y, z, 0, -np.pi/2, 0]

def quarter_circle_velocity(t, center=[0.5, 0.0, 0.4], radius=0.2, frequency=1.0):
    omega = (np.pi / 2) * frequency
    theta = (3 * np.pi / 2) + (np.pi / 2) * (frequency * t % 1.0)
    dx = -radius * omega * np.sin(theta)
    dy = 0
    dz = radius * omega * np.cos(theta)
    return [dx, dy, dz, 0, 0, 0]



def get_trajectory_functions(config):
    traj_type = config["trajectory"]["type"]

    if traj_type == "quarter_circle":
        p = config["trajectory"]["quarter_circle"]
        return (
            lambda t: quarter_circle_trajectory(t, p["center"], p["radius"], p["frequency"]),
            lambda t: quarter_circle_velocity(t, p["center"], p["radius"], p["frequency"])
        )

