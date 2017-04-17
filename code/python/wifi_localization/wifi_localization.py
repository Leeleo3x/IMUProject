import numpy as np
import os
import sys

args = None

class WifiDatabase:
    def __init__(self, wifi_records):
        wifi_all = []
        median_time = []
        for scan in wifi_records:
            scan_times = np.array([v['t'] for v in scan], dtype=int)
            median_time.append(np.median(scan_times, axis=0))
            wifi_all += scan

        print('time:', median_time)
        # resort all records by time stamp
        wifi_all = sorted(wifi_all, key=lambda k: k['t'])
        self.bssid_map_ = self.build_bssid_map(wifi_all)
        # re-cluster the scan results with median time stamp
        self.wifi_reordered = [[] for _ in range(len(median_time))]
        for rec in wifi_all:
            cluster_id = np.argmin([abs(rec['t'] - v) for v in median_time])
            self.wifi_reordered[cluster_id].append(rec)

    def build_bssid_map(self, wifi_records):
        return []


def load_wifi_data(path):
    records = []
    with open(path) as wifi_file:
        header = wifi_file.readline()
        num_record = int(wifi_file.readline().strip())
        for i in range(num_record):
            cur_record = []
            num_wifi = int(wifi_file.readline().strip())
            for j in range(num_wifi):
                line = wifi_file.readline().strip().split()
                cur_record.append({'t': int(line[0]), 'BSSID': line[1], 'level': int(line[2])})
            records.append(cur_record)
    return records

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('list')

    args = parser.parse_args()

    root_dir = os.path.dirname(args.list)

    with open(args.list) as f:
        datasets = f.readlines()

    wifi_records = []
    for data in datasets:
        if len(data) == 0:
            continue
        data_name = data.strip().split()[0]
        data_path = root_dir + '/' + data_name
        print('Loading ' + data_path)
        wifi_records += load_wifi_data(data_path + '/wifi.txt')
        print('{} records in the file {}'.format(len(wifi_records), data_name))

    wifi_base = WifiDatabase(wifi_records)
    for sec in wifi_base.wifi_reordered:
        print('----------------------')
        print(len(sec))
        for v in sec:
            print(v)
    print('Total scans in reordered:', len(wifi_base.wifi_reordered))
