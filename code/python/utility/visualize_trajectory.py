import sys
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import argparse
import utility.write_trajectory_to_ply as write_ply

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', type=str)
    args = parser.parse_args()
    poses = np.genfromtxt(args.dir + "/pose.txt")
    orientation = poses[:, -4:]
    orientation[:, [0, 1, 2, 3]] = orientation[:, [3, 0, 1, 2]]
    position = poses[:, 1:4]

    mag_array = []
    current = 0
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(position[:, 0], position[:, 1], position[:, 2])
    ax.legend()
    ax.set_zlim3d(0, 5)
    plt.show()


    # out_path = args.dir + '/trajectory.ply'
    # # To overlap the trajectory with the Tango trajectory, we need to apply a global rotation
    # global_rotation = np.array([[1.0, 0.0, 0.0],
    #                             [0.0, 0.0, -1.0],
    #                             [0.0, 1.0, 0.0]])
    # write_ply.write_ply_to_file(out_path, position, orientation, global_rotation=global_rotation,
    #                             trajectory_color=[255, 0, 0], interval=16, kpoints=0)
    # print('Ply file written to ' + out_path)
