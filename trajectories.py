import numpy as np

def quarter_circle_trajectory(t, center=[0.5, 0.0, 0.4], radius=0.2, frequency=0.1):
    theta = 1.5 * np.pi + 0.5 * np.pi * (frequency * t % 1.0)
    x = center[0] + radius * np.cos(theta)
    y = center[1]
    z = center[2] + radius * np.sin(theta)
    return [x, y, z, 0, 0, 0]

def quarter_circle_velocity(t, center=[0.5, 0.0, 0.4], radius=0.2, frequency=0.1):
    omega = 0.5 * np.pi * frequency
    theta = 1.5 * np.pi + 0.5 * np.pi * (frequency * t % 1.0)
    dx = -radius * omega * np.sin(theta)
    dy = 0
    dz = radius * omega * np.cos(theta)
    return [dx, dy, dz, 0, 0, 0]

def get_trajectory_functions(config):
    center = config["trajectory"]["center"]
    radius = config["trajectory"]["radius"]
    frequency = config["trajectory"]["frequency"]
    return (
        lambda t: quarter_circle_trajectory(t, center, radius, frequency),
        lambda t: quarter_circle_velocity(t, center, radius, frequency)
    )
