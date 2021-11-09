'''
run multiple tasks with a command
usage: 
1. modify the function define_task()
2. call run_tasks()
'''

import os
import json
import subprocess
from glob import glob
from os.path import dirname
from time import sleep

used_devices = []

def exec_cmd(cmd):
    r = os.popen(cmd)
    msg = r.read()
    r.close()
    if 'json' in cmd:
        return json.loads(msg)
    else:
        return msg


def get_device(msg):
    for gpu in msg['gpus']:
        index = gpu['index']
        state = len(gpu['processes'])
        if state == 0 and index not in used_devices:
            return index
    return -1


def run_one_task(cmd, device, log_file):
    log_file_path = 'logs/{}.log'.format(log_file)
    log_path = dirname(log_file_path)
    if not os.path.exists(log_path):
        os.mkdir(log_path)

    complete_cmd = 'CUDA_VISIBLE_DEVICES={} nohup {} > logs/{}.log 2>&1 &'.format(device, cmd, log_file)
    print(complete_cmd)

    used_devices.append(device)
    subprocess.Popen(complete_cmd, shell=True)


def run_tasks(task_pool, prefix='', sleep_time=1):
    number = 0
    while len(task_pool) > 0:
        device = get_device(
            exec_cmd('/home/huchi/anaconda3/envs/openprompt/bin/gpustat --json'))

        current_task_name = task_pool[0].split()[-1].split('/')[-1]
        if len(task_pool) == 0:
            exit(0)
        if device == -1:
            print('GPUs are busy...')
            sleep(sleep_time)
            continue
        elif not os.path.exists('logs/{}.log'.format(current_task_name)):
            if len(task_pool) > 0:
                run_one_task(task_pool[0], device, current_task_name)
                number += 1
                if 'search' not in task_pool[0]:
                    sleep(sleep_time)
                task_pool.pop(0)
                continue
            else:
                exit(0)
        else:
            task_pool.pop(0)
            print('This task is done...')


def define_task():

    tasks = []
    base_cmd = '/home/huchi/anaconda3/envs/openprompt/bin/python -u experiments/cli.py '

    for f in glob('experiments/*.yaml'):
        cmd = base_cmd
        cmd += '--config_yaml {}'.format(f)
        tasks.append(cmd)

    return tasks


tasks = define_task()
run_tasks(tasks, '', 1)
