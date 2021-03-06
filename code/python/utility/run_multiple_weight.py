import sys
import os
import warnings
import subprocess
import argparse

sys.path.append('/home/yanhang/Documents/research/IMUProject/code/python')
sys.path.append('/Users/yanhang/Documents/research/IMUProject/code/python')

parser = argparse.ArgumentParser()
parser.add_argument('list', type=str, default=None)
parser.add_argument('--recompute', action='store_true')
args = parser.parse_args()

exec_path = '../../cpp/cmake-build-relwithdebinfo/imu_localization/IMULocalization_cli'
model_path = '../../../models/svr_cascade1111'

# weight_list = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
weight_list = [0.0001, 10000]
# suffix_list = ['001', '01', '1', '10', '100', '1000', '10000']
suffix_list = ['00001', '10000']

assert len(weight_list) == len(suffix_list)

root_dir = os.path.dirname(args.list)
data_list = []
with open(args.list) as f:
    for line in f.readlines():
        if line[0] == '#':
            continue
        info = line.split(',')
        if len(info) > 0:
            data_list.append(info[0].strip('\n'))

# Sanity check
all_good = True
for data in data_list:
    data_path = root_dir + '/' + data
    if not os.path.isdir(data_path):
        print(data_path + 'does not exist')
        all_good = False

assert all_good, 'Some datasets do not exist. Please fix the data list.'
print('Sanity check passed')

for data in data_list:
    data_path = root_dir + '/' + data
    if not os.path.isdir(data_path):
        warnings.warn(data_path + ' does not exist. Skip.')
        continue
    for i in range(len(weight_list)):
        command = "%s %s --model_path %s --weight_ls %f --suffix %s" % (exec_path, data_path, model_path,
                                                                        weight_list[i], suffix_list[i])
        print(command)
        subprocess.call(command, shell=True)
