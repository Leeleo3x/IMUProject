import sys
import os
import subprocess
import argparse

sys.path.append('/home/yanhang/Documents/research/IMUProject/code/python')
sys.path.append('/Users/yanhang/Documents/research/IMUProject/code/python')

parser = argparse.ArgumentParser()
parser.add_argument('--recompute', action='store_true')
args = parser.parse_args()

exec_path = '../../cpp/cmake-build-relwithdebinfo/imu_localization/IMULocalization_cli'
model_path = '../../../models/svr_cascade1004_c1e001'
data_root1 = '../../../data2/'
data_root2 = '../../../data/phab_body/'
data_list1 = ['hang_handheld_normal2', 'hang_handheld_speed2', 'hang_handheld_side2', 'hang_leg_normal2', 'hang_leg_speed2', 'hang_leg_side2',
              'hang_bag_normal2', 'hang_bag_speed2', 'hang_bag_side2', 'chen_handheld2', 'chen_body2', 'chen_bag2', 'yajie_handheld1', 'yajie_leg1',
              'yajie_bag1', 'yajie_body1', 'huayi_handheld2', 'huayi_leg2']
data_list2 = ['lopata2', 'library1', 'cse1', 'huayi_cse1', 'huayi_cse3']

preset_list = ['full', 'const']