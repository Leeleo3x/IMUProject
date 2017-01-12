import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('dir')

args = parser.parse_args()

pose_data = np.genfromtxt(args.dir+'/pose.txt')
acce_data = np.genfromtxt(args.dir+'/acce.txt')
gyro_data = np.genfromtxt(args.dir+'/gyro.txt')

