import math

import numpy as np
from numba import jit
import quaternion


def rotate_vector(input, orientation):
    output = np.empty(input.shape, dtype=float)
    for i in range(input.shape[0]):
        q = quaternion.quaternion(*orientation[i])
        output[i] = (q * quaternion.quaternion(1.0, *input[i]) * q.conj()).vec
    return output

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

    # output = np.empty(data.shape, dtype=float)
    output = np.copy(data)
    for i in range(data.shape[0]):
        q = quaternion_from_two_vectors(gravity[i], local_g_direction)
        if q.w < 0.99:
            output[i] = (q * quaternion.quaternion(1.0, *data[i]) * q.conj()).vec
    return output


@jit
def adjust_eular_angle(source):
    # convert the euler angle s.t. pitch is in (-pi/2, pi/2), roll and yaw are in (-pi, pi)
    output = np.copy(source)
    if output[0] < -math.pi / 2:
        output[0] += math.pi
        output[1] *= -1
        output[2] += math.pi
    elif output[0] > math.pi / 2:
        output[0] -= math.pi
        output[1] *= -1
        output[2] -= math.pi

    for j in [1, 2]:
        if output[j] < -math.pi:
            output[j] += 2 * math.pi
        if output[j] > math.pi:
            output[j] -= 2 * math.pi
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

    # output = np.empty(data.shape, dtype=float)
    output = np.copy(data)

    # be careful of the ambiguity of eular angle representation
    for i in range(data.shape[0]):
        rotor = quaternion_from_two_vectors(gravity[i], local_g_direction)
        if np.linalg.norm(rotor.vec) > 1e-3 and rotor.w < 0.999:
            q = rotor * quaternion.from_euler_angles(*data[i]) * rotor.conj()
            output[i] = adjust_eular_angle(quaternion.as_euler_angles(q))
        # if np.linalg.norm(output[i] - data[i]) > 0.1:
        #     print('--------------------\n')
        #     print('ori: {:.6f}, {:.6f}, {:.6f}\nout: {:.6f}, {:.6f}, {:.6f}\nrot: {:.6f}, {:.6f}, {:.6f}, {:.6f}\n'
        #           'grv: {:.6f}, {:.6f}, {:.6f}'.format(data[i][0], data[i][1], data[i][2],
        #                                                output[i][0], output[i][1], output[i][2],
        #                                                rotor.w, rotor.x, rotor.y, rotor.z,
        #        gravity[i][0], gravity[i][1], gravity[i][2]))
    return output


def orientation_from_gravity_and_magnet(grav, magnet,
                                        local_gravity=np.array([0, 1, 0]),
                                        global_gravity=np.array([0, 0, 1]),
                                        global_north=np.array([-1, 0, 0])):
    rot_grav = quaternion_from_two_vectors(grav, global_gravity)
    # remove tilting
    rot_tilt = quaternion_from_two_vectors(grav, local_gravity)
    magnet_grav = (rot_tilt * quaternion.quaternion(1.0, *magnet) * rot_tilt.conj()).vec
    magnet_grav[1] = 0.0
    magnet_grav = (rot_grav * quaternion.quaternion(1.0, *magnet_grav) * rot_grav.conj()).vec
    rot_magnet = quaternion_from_two_vectors(magnet_grav, global_north)
    return rot_magnet * rot_grav


if __name__ == '__main__':

    import pandas
    import quaternion

    data_dir = '../../../data/phab_body/cse1'
    data_all = pandas.read_csv(data_dir + '/processed/data.csv')[:-5]
    ts = data_all['time'].values
    position = data_all[['pos_x', 'pos_y', 'pos_z']].values
    orientation = data_all[['ori_w', 'ori_x', 'ori_y', 'ori_z']].values
    gravity = data_all[['grav_x', 'grav_y', 'grav_z']].values
    magnet_data = np.genfromtxt(data_dir + '/magnet.txt')

    from scipy.interpolate import interp1d
    magnet_func = interp1d(magnet_data[:, 0], magnet_data[:, 1:], axis=0)
    magnet = magnet_func(ts)

    ori_grav_mag = []
    for i in range(ts.shape[0]):
        ori_grav_mag.append(orientation_from_gravity_and_magnet(gravity[i], magnet[i]))

    rv = data_all[['rv_w', 'rv_x', 'rv_y', 'rv_z']].values
    init_trans = quaternion.quaternion(*rv[0]) * ori_grav_mag[0].inverse()
    for i in range(ts.shape[0]):
       ori_grav_mag[i] = init_trans * ori_grav_mag[i]

    mag_grav = align_3dvector_with_gravity(magnet, gravity)
    mag_grav[:, 1] = 0

    ori_grav_mag = quaternion.as_float_array(ori_grav_mag)
    from utility import write_trajectory_to_ply
    write_trajectory_to_ply.write_ply_to_file(data_dir + '/processed/test_grav_magnet.ply', position, ori_grav_mag,
                                              acceleration=mag_grav,
                                              interval=200)

    print('Trajectory written')




