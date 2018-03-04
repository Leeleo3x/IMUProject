import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pyquaternion import Quaternion
import argparse
import math
from numpy import linalg as la
from matplotlib import animation


def build_data(magnet, acce):
    mag_array = []
    current = 0
    for i in range(acce.shape[0]):
        if acce[i, 0] <= magnet[current, 0] or current+1 == magnet.shape[0]:
            mag_array.append(magnet[current, 1:4])
        else:
            current += 1
            mag_array.append(magnet[current, 1:4])
    return np.array(mag_array)


def orientation(magnet, acce):
    na = la.norm(acce)
    pitch = np.arcsin(-acce[1]/na)
    roll = np.arcsin(acce[0]/na)
    y = (-magnet[0]) * np.cos(roll) + magnet[2] * np.sin(roll)
    x = magnet[0] * np.sin(pitch) * np.sin(roll) + magnet[1] * np.cos(pitch) + magnet[2] * np.sin(pitch) * np.cos(roll)
    azimuth = np.arctan2(y, x)
    return [azimuth, pitch, roll]


def show(folder, start, num):
    tango_start = 1109611481633654
    pixel_start = 121901693862541
    fig = plt.figure(num)
    def move(df):
        df[:, [4, 1, 2, 3]] = df[:, [1, 2, 3, 4]]
        return df
    # poses = np.genfromtxt(folder+"/tango/pose.txt")
    rotation = move(np.genfromtxt(folder + "/tango/orientation.txt"))
    pixel = move(np.genfromtxt(folder + "/pixel/orientation.txt"))
    position = np.genfromtxt(folder+"/tango/pose.txt")[:, 0:4]

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlim3d([-4.5, 4.5])
    ax.set_xlabel('X')
    ax.set_ylim3d([-4.5, 4.5])
    ax.set_ylabel('Y')
    ax.set_zlim3d([-4.5, 4.5])
    ax.set_zlabel('Z')
    # ax.plot(position[:, 1], position[:, 2], position[:, 3])
    init = np.array(
        [0, 0, 0]
    )
    line0 = ax.plot([0], [0], [0], 'r')[0]
    line1 = ax.plot([0], [0], [0], 'g')[0]
    line2 = ax.plot([0], [0], [0], 'b')[0]
    ax.legend()

    quaternion_pixel = []
    quaternion_tango = []
    angle = []
    dir_pixel = []
    dir_tango = []
    v = np.array([1, 0, 0])
    for x in pixel:
        print((x[0] - pixel_start)/1000000)
        if (x[0] - pixel_start) / 1000000 > 0:
            quaternion_pixel.append(Quaternion(x[1:5]))
            dir_pixel.append(quaternion_pixel[-1].rotate(v))
    for x in rotation:
        if (x[0] - tango_start) / 1000000 > 0:
            quaternion_tango.append(Quaternion(x[1:5]))
            dir_tango.append(quaternion_tango[-1].rotate(v))


    for i in range(min(len(quaternion_pixel), len(quaternion_tango))):
        x = quaternion_pixel[i].rotate(v)
        y = quaternion_tango[i].rotate(v)
        res = np.dot(x, y)
        angle.append(math.acos(res) / math.pi * 180)

    ppp = []
    pre = 0
    current = -1
    for x in position:
        idx = int((x[0] - tango_start) / 5000000)
        if idx < 0:
            continue
        if idx != current:
            current = idx
            ppp.append(x[1:4]*0.1-np.array([0.5, 1.3, 0.5]))

    ppp = np.array(ppp)
    ppp[:, 0] *= -1
    ppp = ppp.T

    def insert(i):
        x = dir_pixel[i]
        y = dir_tango[i]
        point = np.array([0, 0, 0])
        idx = np.concatenate((point.T, (y + point.T)), axis=0)
        idx = idx.reshape(-1, 3).T
        idx2 = np.concatenate((point.T, (x + point.T)), axis=0)
        idx2 = idx2.reshape(-1, 3).T
        line1.set_data(idx[0:2, :])
        line1.set_3d_properties(idx[2, :])
        line0.set_data(idx2[0:2, :])
        line0.set_3d_properties(idx2[2, :])
        line2.set_data(ppp[0:2, :i])
        line2.set_3d_properties(ppp[2, :i])
        return [line0, line1, line2]

    anim = animation.FuncAnimation(fig, insert, frames=min(len(dir_pixel), len(dir_tango)), interval=5, blit=True)
    plt.figure(1)
    plt.plot(angle)

    # anim.save('example.mp4', fps=30)
    plt.show()



    def view(array):
        array = np.array(array)
        plt.plot(array[:, 0], 'r', array[:, 1], 'g', array[:, 2], 'b')

    plt.figure(num+1)
    view(dir_pixel)
    plt.figure(num+2)
    view(dir_tango)



        # v = np.array([0, 0, 1])
        # px = Quaternion(rotation[i, 1:5])
        # py = Quaternion(pixel[i, 1:5])
        # x = px.rotate(v)
        # y = py.rotate(v)
        # res = np.dot(x, y)
        # angle.append(math.acos(res) / math.pi * 180)


    # new_pos = 0
    # for i in range(poses.shape[0]):
    #     if poses[i, 0] >= tango_start:
    #         new_pos = i
    #         break
    # print(new_pos)
    # poses = poses[new_pos:, :]
    #
    # current = 0
    # current_pixel = 0
    # new_rotation = []
    # new_pixel = []
    # angle = []
    # for i in range(poses.shape[0]):
    #     v = np.array([0, 0, 1])
    #     tango_current = poses[i, 0] - tango_start
    #     while current_pixel+1 < pixel.shape[0] and pixel[current_pixel+1, 0] - pixel_start < tango_current:
    #         current_pixel += 1
    #     if pixel[current_pixel+1, 0] - pixel_start == tango_current:
    #         qp = Quaternion(pixel[tango_current+1, 1:5])
    #         new_pixel.append(Quaternion(pixel[tango_current+1, 1:5]).rotate(v))
    #     else:
    #         q1 = Quaternion(pixel[current_pixel, 1:5])
    #         q2 = Quaternion(pixel[current_pixel+1, 1:5])
    #         qp = Quaternion.slerp(q1, q2, (tango_current - pixel[current_pixel, 0] + pixel_start) /
    #                              (pixel[current_pixel+1, 0]) - pixel[current_pixel, 0])
    #         new_pixel.append(qp.rotate(v))
    #
    #     while current+1 < rotation.shape[0] and rotation[current+1, 0] < poses[i, 0]:
    #         current += 1
    #     if rotation[current+1, 0] == poses[i, 0]:
    #         qt = Quaternion(rotation[current+1, 1:5])
    #         new_rotation.append(Quaternion(rotation[current+1, 1:5]).rotate(v))
    #     else:
    #         q1 = Quaternion(rotation[current, 1:5])
    #         q2 = Quaternion(rotation[current+1, 1:5])
    #         q = Quaternion.slerp(q1, q2, (poses[i, 0] - rotation[current, 0]) / (rotation[current + 1, 0] -
    #                                                                              rotation[current, 0]))
    #         qt = q
    #         new_rotation.append(q.rotate(v))
    #     res = np.dot(new_pixel[-1], new_rotation[-1])
    #     angle.append(math.acos(res) / math.pi * 180)


    # start = 5000
    # end = 10 + start
    # position = poses[:, 1:4]
    # position *= 1000
    # ax = fig.gca(projection='3d')
    # ax.plot(position[start:end, 0], position[start:end, 1], position[start:end, 2])
    # new_rotation = np.array(new_rotation)
    # new_rotation += position
    # new_pixel = np.array(new_pixel)
    # new_pixel += position
    # ax.quiver(position[start:end, 0], position[start:end, 1], position[start:end, 2], new_rotation[start:end, 0],
    #           new_rotation[start:end, 1], new_rotation[start:end, 2], normalize=True, arrow_length_ratio=0.1)
    # ax.quiver(position[start:end, 0], position[start:end, 1], position[start:end, 2], new_pixel[start:end, 0],
    #           new_pixel[start:end, 1], new_pixel[start:end, 2], normalize=True, arrow_length_ratio=0.1)
    # ax.legend()
    # fig = plt.figure(num+1)
    # new_rotation -= position
    # new_pixel -= position
    # print(new_pixel.shape)
    # print(position.shape)
    # print(new_rotation.shape)
    # plt.plot(new_rotation[:, 0], 'r', new_pixel[:, 0], 'r')
    # plt.plot(new_rotation[:, 1], 'g', new_pixel[:, 1], 'g')
    # plt.plot(new_rotation[:, 2], 'b', new_pixel[:, 2], 'b')
    # fig = plt.figure(num+2)


def diff(d1, d2):
    s1 = 0
    s2 = 0
    diff = []
    while s1 < d1.shape[0] and s2 < d2.shape[0]:
        diff.append((d1[s1, 1] - d2[s2, 1]) / math.pi * 180)
        if d1[s1, 0] < d2[s2, 0]:
            s1 += 1
        elif d1[s1, 0] > d2[s2, 0]:
            s2 += 1
        else:
            s1 += 1
            s2 += 1
    plt.figure(100)
    plt.plot(diff, 'r')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', type=str)
    args = parser.parse_args()
    # tasc - 4
    # 2280 2256
    # tasc - 3
    # 1959 - 2071
    # show(os.path.join(args.dir, 'pixel'), 0, 0)
    show(args.dir, 0, 1)
    # diff(r1, r2)
    plt.show()
