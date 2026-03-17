import numpy as np
import os
    
script_path = os.path.abspath(__file__)
script_directory = os.path.dirname(script_path)

filtered_file = os.path.join(script_directory, 'interpolated.csv')
inocean = os.path.join(script_directory, 'filtered_drillship.csv')

def quaternion_to_euler(x, y, z, w):
    """
    Convert quaternion (x, y, z, w) to Euler angles (roll, pitch, yaw)
    in radians.
    """
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if np.abs(sinp) >= 1:
        pitch = np.sign(sinp) * np.pi / 2  # Use 90 degrees if out of range
    else:
        pitch = np.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw

def body_to_world(body_vel_x, body_vel_y, yaw):
    """
    Rotate body-frame velocities to world-frame using yaw angle (ψ).
    :param body_vel_x: forward velocity in body frame
    :param body_vel_y: lateral velocity in body frame
    :param yaw: yaw angle in radians
    :return: vx_world, vy_world
    """
    vx_world = body_vel_x * np.cos(yaw) - body_vel_y * np.sin(yaw)
    vy_world = body_vel_x * np.sin(yaw) + body_vel_y * np.cos(yaw)
    measure = np.linalg.norm(np.array([vx_world, vy_world]))
    return vx_world, vy_world, measure