import math
import numpy as np
from reachability_utils.plot_reachable_sets import convert_A_b_to_Q_c_r
from scipy.optimize import minimize

def rotation_matrix(angle, radian=False):
    """Create a 2D rotation matrix

    :param angle: angle in degrees or radians
    :param radian: boolean flag indicating if the angle is in radians
    """
    rad = degree_to_radian(angle) if not radian else angle
    return np.array([[math.cos(rad), -math.sin(rad)], [math.sin(rad), math.cos(rad)]])

def normalize_degree(degree):
    """Normalize degrees to the range [0, 360]

    :param deg: degrees
    :return: normalized degrees
    """
    return degree % 360


def normalize_radian(radian):
    """Normalize radians to the range [0, 2*pi]

    :param rad: radians
    :return: normalized radians
    """
    return radian % (2 * math.pi)

def normalize_radian_pi(radian):
    """
    Normalize angle to be between -pi and pi.
    """
    return (radian + np.pi) % (2 * np.pi) - np.pi


def degree_to_radian(degree):
    """Convert degrees to radians

    :param deg: degrees
    :return: radians
    """
    return degree * math.pi / 180


def radian_to_degree(radian):
    """Convert radians to degrees

    :param rad: radians
    :return: degrees
    """
    return radian * 180 / math.pi