import argparse
import subprocess
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('list', type=str)
    parser.add_argument('command', type=str)

    args = parser.parse_args()

    with open(args.list) as f:
        datasets = [data.strip('\n') for data in f.readlines()]

    root_dir = os.path.dirname(args.list)
    for data in datasets:
        if len(data) == 0:
            continue
        if data[0] == '#':
            continue
        info = data.split(',')
        command = 'cd {}/{} && {}'.format(root_dir, info[0], args.command)
        print(command)
        subprocess.call(command, shell=True)