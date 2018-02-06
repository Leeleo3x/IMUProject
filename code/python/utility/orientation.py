import numpy as np
import argparse
import math
import os
from numpy import linalg as la
import matplotlib.pyplot as plt


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
    plt.figure(num)
    acce = np.genfromtxt(folder + "/acce.txt")
    plt.plot(acce[:, 1], 'r', acce[:, 2], 'g', acce[:, 3], 'b')


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
    show(os.path.join(args.dir, 'pixel'), 0, 0)
    show(os.path.join(args.dir, 'tango'), 0, 1)
    # diff(r1, r2)
    plt.show()
