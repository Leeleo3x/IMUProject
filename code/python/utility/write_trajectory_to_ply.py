import numpy as np
import plyfile
import quaternion


def write_ply_to_file(path, position, orientation,
                      length = 1.0, kpoints=200):
    """
    Visualize camera trajectory as ply file
    :param path: path to save
    :param position: Nx3 array of positions
    :param orientation: Nx4 array or orientation as quaternion
    :return: None
    """
    num_cams = position.shape[0]
    assert orientation.shape[0] == num_cams

    sample_pt = np.arange(0, num_cams, 300, dtype=int)
    num_sample = sample_pt.shape[0]

    # local coordinate system is computed by
    # local_x = quaternion.quaternion(0.0, 1.0, 0.0, 0.0)
    # local_y = quaternion.quaternion(0.0, 0.0, 0.0, -1.0)
    # local_z = quaternion.quaternion(0.0, 0.0, 1.0, 0.0)
    local_axis = np.array([[1.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0],
                          [0.0, 0.0, 1.0]])

    local_axis2 = np.array([[1.0, 0.0, 0.0],
                           [0.0, 0.0, 1.0],
                           [0.0, -1.0, 0.0]])

    # first compute three axis direction as unit vector in global frame
    # glob_ori_x = np.empty([num_sample, 3], dtype=float)
    # glob_ori_y = np.empty([num_sample, 3], dtype=float)
    # glob_ori_z = np.empty([num_sample, 3], dtype=float)

    axis_color = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
    vertex_type = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    positions_data = np.empty((position.shape[0],), dtype=vertex_type)
    positions_data[:] = [tuple([*i, 0, 255, 255]) for i in position]

    # temporal array to store axis vertices at each sampled location
    # global_axes = np.empty([3, 3], dtype=float)
    print(local_axis2)
    rm = quaternion.as_rotation_matrix(quaternion.quaternion(*orientation[1000]))
    print(rm)
    print(np.dot(local_axis2, rm))
    rm[:, [1, 2]] = rm[:, [2, 1]]
    rm[:, 1] *= -1
    print(rm)

    app_vertex = np.empty([3 * kpoints], dtype=vertex_type)
    for i in range(num_sample):
        q = quaternion.quaternion(*orientation[sample_pt[i]])
        # global_axes = np.matmul(quaternion.as_rotation_matrix(q), local_axis)
        # global_axes = np.matmul(local_axis2, quaternion.as_rotation_matrix(q))
        global_axes = quaternion.as_rotation_matrix(q)
        # global_axes[:, [1, 2]] = global_axes[:, [2,1 ]]
        # global_axes[:, 1] *= -1
        # global_axes[0] = (q * local_x * q.conj()).vec
        # global_axes[1] = (q * local_y * q.conj()).vec
        # global_axes[2] = (q * local_z * q.conj()).vec

        for k in range(3):
            for j in range(kpoints):
                axes_pts = position[sample_pt[i]].flatten() + global_axes[:, k].flatten() * j * length / kpoints
                app_vertex[k*kpoints + j] = tuple([*axes_pts, *axis_color[k]])

        positions_data = np.concatenate([positions_data, app_vertex], axis=0)
    vertex_element = plyfile.PlyElement.describe(positions_data, 'vertex')
    plyfile.PlyData([vertex_element], text=True).write(path)


if __name__ == '__main__':
    import argparse
    import pandas

    parser = argparse.ArgumentParser()
    parser.add_argument('dir', type=str)
    parser.add_argument('output', type=str)

    args = parser.parse_args()

    data_all = pandas.read_csv(args.dir + '/processed/data.csv')
    orientation = data_all[['ori_w', 'ori_x', 'ori_y', 'ori_z']].values
    position = data_all[['pos_x', 'pos_y', 'pos_z']].values

    print('Writing ply file')
    write_ply_to_file(path=args.output, position=position, orientation=orientation)
    print('File writing to ' + args.output)
