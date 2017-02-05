import numpy as np
import plyfile
import quaternion


def write_ply_to_file(path, position, orientation, acceleration = None, length = 1.0, kpoints=200, interval=100):
    """
    Visualize camera trajectory as ply file
    :param path: path to save
    :param position: Nx3 array of positions
    :param orientation: Nx4 array or orientation as quaternion
    :param acceleration: (option) Nx3 array of acceleration
    :return: None
    """
    num_cams = position.shape[0]
    assert orientation.shape[0] == num_cams

    num_axis = 3
    max_acceleration = 1.0
    if acceleration is not None:
        assert acceleration.shape[0] == num_cams
        max_acceleration = max(np.linalg.norm(acceleration, axis=1))
        print('max_acceleration: ', max_acceleration)
        num_axis = 4

    sample_pt = np.arange(0, num_cams, interval, dtype=int)
    num_sample = sample_pt.shape[0]

    # local coordinate system is computed by
    imu_to_tango = np.array([[1.0, 0.0, 0.0],
                             [0.0, 0.0, 1.0],
                             [0.0, -1.0, 0.0]])

    axis_color = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 0, 255]]
    vertex_type = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    positions_data = np.empty((position.shape[0],), dtype=vertex_type)
    positions_data[:] = [tuple([*i, 0, 255, 255]) for i in position]

    local_axis = np.array([[1.0, 0.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0, 0.0],
                           [0.0, 0.0, 1.0, 0.0]])
    app_vertex = np.empty([num_axis * kpoints], dtype=vertex_type)
    for i in range(num_sample):
        q = quaternion.quaternion(*orientation[sample_pt[i]])
        if acceleration is not None:
            local_axis[:, -1] = acceleration[sample_pt[i]].flatten() / max_acceleration

        global_axes = np.matmul(quaternion.as_rotation_matrix(q), local_axis)
        # if i == 0:
        #     print('-------------\nwrite_ply_file')
        #     print('rotation\n', quaternion.as_rotation_matrix(q))
        #     print('acceleration:', acceleration[sample_pt[i]])
        #     print('local axis: ', local_axis[:, -1])
        #     print(global_axes[:, 3])
        for k in range(num_axis):
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
    linacce = data_all[['linacce_x', 'linacce_y', 'linacce_z']].values
    gravity = data_all[['grav_x', 'grav_y', 'grav_z']].values
    position = data_all[['pos_x', 'pos_y', 'pos_z']].values
    gravity *= -1

    print(linacce[0:500:20])
    print(np.linalg.norm(linacce[0:500:20], axis=1))

    print('Writing ply file')
    write_ply_to_file(path=args.output, position=position, orientation=orientation, acceleration=linacce, interval=20)
    print('File writing to ' + args.output)
