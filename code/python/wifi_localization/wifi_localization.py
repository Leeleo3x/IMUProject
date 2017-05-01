import numpy as np
import os
import sys
import json
sys.path.append('/Users/yanhang/Documents/research/IMUProject/code/python')
sys.path.append('/home/yanhang/Documents/research/IMUProject/code/python')

from pre_processing import gen_dataset

args = None

micro_to_nano = 1000


def reorder_wifi_records(wifi_records):
    median_time = []
    for scan in wifi_records:
        scan_times = []
        for rec in scan:
            scan_times.append(rec['t'])
        median_time.append(np.median(scan_times, axis=0))
    # re-cluster the scan results with median time stamp
    wifi_reordered = [[] for _ in range(len(median_time))]
    for scan in wifi_records:
        for rec in scan:
            cluster_id = np.argmin([abs(rec['t'] - v) for v in median_time])
            is_new = True
            for ext in wifi_reordered[cluster_id]:
                if ext['BSSID'] == rec['BSSID']:
                    ext['level'] = max(rec['level'], ext['level'])
                    is_new = False
                    break
            if is_new:
                wifi_reordered[cluster_id].append(rec)
    # remove empty scans
    for scan in wifi_reordered:
        if len(scan) == 0:
            wifi_reordered.remove(scan)
    return wifi_reordered


def build_bssid_map(wifi_records, min_count=5):
    bssid_count = {}
    for scan in wifi_records:
        for rec in scan:
            if rec['BSSID'] not in bssid_count:
                bssid_count[rec['BSSID']] = 1
            else:
                bssid_count[rec['BSSID']] += 1
    bssid_map = {}
    ind = 0
    for scan in wifi_records:
        for rec in scan:
            if bssid_count[rec['BSSID']] < min_count:
                continue
            elif rec['BSSID'] not in bssid_map:
                bssid_map[rec['BSSID']] = ind
                ind += 1
    return bssid_map


def filter_scan(scan, min_time, max_time, min_level):
    scan_filtered = []
    for rec in scan:
        if min_time < rec['t'] < max_time and rec['level'] > min_level:
            scan_filtered.append(rec)
    return scan_filtered


def merge_grouped_records(wifi_records, grouping=1):
    assert len(wifi_records) % grouping == 0
    merged_records = [[] for _ in range(len(wifi_records) // grouping)]
    for gid in range(len(merged_records)):
        for scan in wifi_records[gid * grouping:(gid+1) * grouping]:
            for rec in scan:
                append_new = True
                if len(merged_records[gid]) > 0:
                    for ext_rec in merged_records[gid]:
                        if rec['BSSID'] == ext_rec['BSSID']:
                            append_new = False
                            ext_rec['level'] = max(ext_rec['level'], rec['level'])
                if append_new:
                    merged_records[gid].append(rec)
    return merged_records


def downsample_grouped_records(wifi_records, grouping=1):
    ds = []
    for i in range(0, len(wifi_records), grouping):
        ds.append(wifi_records[i])
    return ds


def build_wifi_footprint(scan, bssid_map, min_level=-100):
    assert len(scan) > 0
    footprint = [min_level for _ in range(len(bssid_map))]
    position = None
    if 'pos' in scan[0]:
        position = sum([v['pos'] for v in scan]) / float(len(scan))
    for rec in scan:
        if rec['BSSID'] in bssid_map:
            footprint[bssid_map[rec['BSSID']]] = rec['level']
    return np.array(footprint), position


def load_wifi_data(path):
    records = []
    with open(path) as wifi_file:
        header = wifi_file.readline()
        redun = int(wifi_file.readline().strip())
        num_record = int(wifi_file.readline().strip())
        for i in range(num_record):
            cur_record = []
            num_wifi = int(wifi_file.readline().strip())
            for j in range(num_wifi):
                line = wifi_file.readline().strip().split()
                level = int(line[2])
                cur_record.append({'t': int(line[0]) * micro_to_nano, 'BSSID': line[1], 'level': level})
            records.append(cur_record)
    return records, redun


def write_wifi_footprints(footprints, bssid_map, path, positions=None):
    json_obj = {'footprints': footprints.tolist(), 'bssid_map': bssid_map}
    if positions is not None:
        json_obj['positions'] = positions.tolist()
    with open(path, 'w') as f:
        json.dump(json_obj, f)


def read_wifi_foorprints(path):
    footprints = None
    bssid_map = {}
    positions = None
    with open(path, 'r') as f:
        json_obj = json.load(f)
        footprints = np.array(json_obj['footprints'])
        bssid_map = json_obj['bssid_map']
        if 'positions' in json_obj:
            positions = np.array(json_obj['positions'])
    return footprints, bssid_map, positions


def query_position(scan, footprints, positions, bssid_map, k=3):
    assert len(footprints) == len(positions)
    assert len(footprints) >= k
    query_footprint, _ = build_wifi_footprint(scan, bssid_map)
    distances = []
    for i in range(len(footprints)):
        dis = np.sort(query_footprint - footprints[i], axis=0)
        distances.append({'id': i, 'dis': np.linalg.norm(dis, ord=2)})
    distances = sorted(distances, key=lambda v: v['dis'])
    query_pos = np.zeros(3, dtype=float)
    for i in range(k):
        query_pos += positions[distances[i]['id']]
    return query_pos / k, query_footprint

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('list')
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--min_level', default=-75, type=int)
    parser.add_argument('--min_count', default=10, type=int)
    parser.add_argument('--merge_mode', default='none', type=str)
    args = parser.parse_args()

    root_dir = os.path.dirname(args.list)

    with open(args.list) as f:
        datasets = f.readlines()

    wifi_all = []
    for data in datasets:
        if len(data) == 0:
            continue
        data_name = data.strip().split()[0]
        data_path = root_dir + '/' + data_name
        print('Loading ' + data_path)
        wifi_records, redun = load_wifi_data(data_path + '/wifi.txt')
        print('{} scans in file {}'.format(len(wifi_records), data_path))
        wifi_reordered = reorder_wifi_records(wifi_records)
        wifi_records = None
        pose_data = np.genfromtxt(data_path + '/pose.txt')[:, :4]
        # remove records that are out of pose data's time range, or with too small signal level
        wifi_reordered = [filter_scan(scan, pose_data[0][0], pose_data[-1][0], args.min_level)
                          for scan in wifi_reordered]
        if args.merge_mode == 'merge':
            wifi_reordered = merge_grouped_records(wifi_reordered, redun)
        elif args.merge_mode == 'downsample':
            wifi_reordered = downsample_grouped_records(wifi_reordered, redun)

        wifi_with_pose = []
        for scan in wifi_reordered:
            if len(scan) == 0:
                continue
            scan_times = np.array([v['t'] for v in scan])
            rec_poses = gen_dataset.interpolate_3dvector_linear(pose_data, scan_times)
            for i in range(len(rec_poses)):
                scan[i]['pos'] = rec_poses[i][1:]
            wifi_with_pose.append(scan)

        # for scan in wifi_reordered:
        #     print('----------------------')
        #     print(len(scan))
        #     for v in sec:
        #         print(v)
        print('Total scans in reordered:', len(wifi_reordered))
        wifi_all += wifi_with_pose

    bssid_map = build_bssid_map(wifi_all, args.min_count)
    print('{} different BSSIDs'.format(len(bssid_map)))

    print('Constructing footprints...')
    footprints_all = np.empty([len(wifi_all), len(bssid_map)], dtype=int)
    positions_all = np.empty([len(wifi_all), 3])
    for i in range(len(wifi_all)):
        footprint, position = build_wifi_footprint(wifi_all[i], bssid_map)
        footprints_all[i] = footprint
        positions_all[i] = position

    # test self validation
    for i in range(len(wifi_all)):
        pos, _, = query_position(wifi_all[i], footprints_all, positions_all, bssid_map, 1)
        print('{}, ({}, {}, {}) | ({}, {}, {})'.format(i, positions_all[i][0], positions_all[i][1], positions_all[i][2],
                                                       pos[0], pos[1], pos[2]))

    if args.output is not None:
        print('Writing to ' + args.output)
        write_wifi_footprints(footprints_all, bssid_map, args.output, positions_all)
