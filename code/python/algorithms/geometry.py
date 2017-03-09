import math

import numpy as np
import quaternion


def rotation_matrix_from_two_vectors(v1, v2):
    """
    Using Rodrigues rotation formula
    https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    :param v1: starting vector
    :param v2: ending vector
    :return 3x3 rotation matrix
    """
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)
    theta = np.dot(v1, v2)
    if theta == 1:
        return np.identity(3)
    if theta == -1:
        raise ValueError
    k = np.cross(v1, v2)
    k /= np.linalg.norm(k)
    K = np.matrix([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.identity(3) + math.sqrt(1 - theta * theta) * K + np.dot((1 - theta) * K * K, v1)


def quaternion_from_two_vectors(v1, v2):
    """
    Compute quaternion from two vectors
    :param v1:
    :param v2:
    :return Quaternion representation of rotation between v1 and v2
    """
    v1n = v1 / np.linalg.norm(v1)
    v2n = v2 / np.linalg.norm(v2)
    w = np.cross(v1n, v2n)
    q = np.array([1.0 + np.dot(v1n, v2n), *w])
    q /= np.linalg.norm(q)
    return quaternion.quaternion(*q)


def align_with_gravity(poses, gravity, local_g_direction=np.array([0, 0, -1])):
    """
    Adjust pose such that the gravity is at $target$ direction
    @:param poses: N x 7 array, each row is position + orientation (quaternion). The array will be modified in place.
    @:param gravity: real gravity direction
    @:param local_g_direction: z direction before alignment
    @:return None.
    """
    assert poses.ndim == 2, 'Expect 2 dimensional array input'
    assert poses.shape[1] == 7, 'Expect Nx7 array input'
    rotor = quaternion_from_two_vectors(local_g_direction, gravity)
    for pose in poses:
        distance = np.linalg.norm(pose[0:3])
        position_n = pose[0:3] / distance
        pose[0:3] = distance * (rotor * quaternion.quaternion(0.0, *position_n) * rotor.conjugate()).vec
        pose[-4:] = quaternion.as_float_array(rotor * quaternion.quaternion(*pose[-4:]) * rotor.conjugate())