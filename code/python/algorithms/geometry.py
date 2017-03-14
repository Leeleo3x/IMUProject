import math

import numpy as np
from numba import jit
import quaternion

@jit
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


# @jit
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


def align_3dvector_with_gravity(data, gravity, local_g_direction=np.array([0, 1, 0])):
    """
    Adjust pose such that the gravity is at $target$ direction
    @:param data: N x 3 array
    @:param gravity: real gravity direction
    @:param local_g_direction: z direction before alignment
    @:return
    """
    assert data.ndim == 2, 'Expect 2 dimensional array input'
    assert data.shape[1] == 3, 'Expect Nx3 array input'
    assert data.shape[0] == gravity.shape[0], '{}, {}'.format(data.shape[0], gravity.shape[0])

    output = np.empty(data.shape, dtype=float)
    for i in range(data.shape[0]):
        q = quaternion_from_two_vectors(gravity[i], local_g_direction)
        output[i] = (q * quaternion.quaternion(1.0, *data[i]) * q.conj()).vec

    return output


@jit
def adjust_eular_angle(source, target, max_v=math.pi/2):
    # The eular axes might be flipped. Try convert to the original direction
    # The heuristics used here are that 1. the values should be small. 2. the y direction should be same as input
    assert source.shape == target.shape
    output = np.copy(source)
    for i in range(source.shape[0]):
        sign = 1.0
        if output[i][1] > max_v:
            output[i][1] -= math.pi
            sign *= -1
        elif output[i][1] < -max_v:
            output[i][1] += math.pi
            sign *= -1
        if output[i][1] * target[i][1] < 0:
            output[i][1] *= -1
            sign *= -1

        for j in [0, 2]:
            if output[i][j] > max_v:
                output[i][j] = (output[i][j] - math.pi) * -1.0
            elif output[i][j] < -max_v:
                output[i][j] = (output[i][j] + math.pi) * -1.0
            output[i][j] *= sign
    return output


def align_eular_rotation_with_gravity(data, gravity, local_g_direction=np.array([0, 1, 0])):
    """
    Transform the coordinate frame of orientations such that the gravity is aligned with $local_g_direction
    :param data: input orientation in Eular
    :param gravity:
    :param local_g_direction:
    :return:
    """
    assert data.shape[1] == 3, 'Expect Nx3 array'
    assert data.shape[0] == gravity.shape[0], '{}, {}'.format(data.shape[0], gravity.shape[0])

    output = np.empty(data.shape, dtype=float)

    # be careful of the ambiguity of eular angle representation
    for i in range(data.shape[0]):
        rotor = quaternion_from_two_vectors(gravity[i], local_g_direction)
        q = rotor * quaternion.from_euler_angles(*data[i]) * rotor.conj()
        output[i] = quaternion.as_euler_angles(q)

    return adjust_eular_angle(output, data)


if __name__ == '__main__':

    import pandas
    import quaternion

    data_all = pandas.read_csv('../../../data/phab_body/cse1/processed/data.csv')
    
    gyro = data_all[['gyro_x', 'gyro_y', 'gyro_z']].values[1057:1058]

    gyro2 = np.empty(gyro.shape, dtype=float)
    for i in range(gyro.shape[0]):
        gyro2[i] = quaternion.as_euler_angles(quaternion.from_euler_angles(*gyro[i]))

    gyro2 = adjust_eular_angle(gyro2, gyro, math.pi * 0.7)
    diff = 0

    for i in range(0, gyro.shape[0]):
        print('{:.6f}, {:.6f}, {:.6f} | {:.6f}, {:.6f}, {:.6f}'.format(
        gyro[i][0], gyro[i][1], gyro[i][2], gyro2[i][0], gyro2[i][1], gyro2[i][2]))
        cur_diff = np.linalg.norm(gyro2[i] - gyro[i])

        if cur_diff > 1e-9:
            print('diff')
        diff += cur_diff
    print('Diff: ', diff)






