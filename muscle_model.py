import numpy as np

def bicep_force_profile(joint_angle_rad, joint_velocity_rad_s, *args, **kwargs):
    """
    Estimate biceps force during a curl based on elbow joint angle and velocity.
    
    Args:
        joint_angle_rad: Elbow joint angle in radians (0 = extended, ~2.36 = flexed)
        joint_velocity_rad_s: Angular velocity in rad/s
    
    Returns:
        3D force vector (Nx, Ny, Nz) in end-effector frame or world frame
    """
    # Normalize angle for force-length curve (0: extended, 1: fully flexed)
    angle_norm = joint_angle_rad / (3 * np.pi / 4)
    angle_norm = np.clip(angle_norm, 0.0, 1.0)

    # Gaussian peak at ~90° (optimal bicep length)
    f_length = np.exp(-((angle_norm - 0.66) ** 2) / 0.01)

    # Concentric vs. eccentric scaling
    f_velocity = 1.2 if joint_velocity_rad_s < 0 else 0.8

    # Total magnitude
    max_force = kwargs.get("max_force", 100.0)
    magnitude = max_force * f_length * f_velocity

    # Realistic direction: biceps mainly pulls in -Z with small backward (-X) and inward (+Y) bias
    force_direction = np.array([-0.1, 0.05, -0.99])
    force_direction /= np.linalg.norm(force_direction)

    return magnitude * force_direction
