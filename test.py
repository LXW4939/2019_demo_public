#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

import os
import time, json
import numpy as np
import pymongo
import re

"""
    获取某个进程的CPU相关信息
    入参：pid 进程号
    返回值：pid, cpu_usage, mem_usage
"""


def get_cpu_resource_usage_by_pid(pid):
    cpu_usage = 0.0
    mem_usage = 0.0
    os.environ['pid'] = pid
    try:
        usagestring = os.popen('ps aux | grep $pid | awk \'NR==1{print $2,$3,$4}\'').read()
        pid, cpu_usage, mem_usage = usagestring.replace('\n', '').split(" ")
        print("cpu id info:", pid, cpu_usage, mem_usage)
    except BaseException as e:
        print(str(e))
    finally:
        return pid, cpu_usage, mem_usage

"""
    获取某个进程的GPU相关信息
    入参:pid
    返回值:gpu_index,pid,gpu_usage,mem_usage,pid_name
"""


def get_gpu_resource_usage_by_pid(pid):
    os.environ['pid'] = pid
    gpu_index = 0
    gpu_usage = 0.0
    mem_usage = 0.0
    pid_name = 'unknown'
    try:
        usagestring = os.popen('nvidia-smi pmon -c 1| grep $pid | awk \'{print $1,$2,$4,$5,$8}\'').read()
        gpu_index, pid, gpu_usage, mem_usage, pid_name = usagestring.replace('\n','').split(" ")
        print("gpu id info:", gpu_index, pid, gpu_usage, mem_usage, pid_name)
    except BaseException as e:
        print(str(e))
    finally:
        return pid, gpu_index, gpu_usage, mem_usage, pid_name


"""
    获取GPU相关信息
    入参:gpu_index
    返回值:gpu_index,pwr,gtemp,gpu_usage,mem_usage,mclk,pclk
"""


def get_gpu_resource_usage_by_gpu_index(gpu_index):
    pwr = 0
    gtemp = 0
    gpu_usage = 0.0
    mem_usage = 0.0
    mclk = 0
    pclk = 0
    try:
        nr = str(gpu_index + 3)
        usagestring = os.popen('nvidia-smi dmon -c 1  |  awk \'NR=='+nr+'{print $2,$3,$5,$6,$9,$10}\'').read()
        #print("gpu index usagestring info : ", usagestring)
        pwr, gtemp, gpu_usage, mem_usage, mclk, pclk = usagestring.replace('\n', '').split(" ")
        print("gpu index info:", gpu_index, pwr, gtemp, gpu_usage, mem_usage, mclk, pclk)
    except BaseException as e:
        print(str(e))
    finally:
        return gpu_index, pwr, gtemp, gpu_usage, mem_usage, mclk, pclk


"""
    获取某个GPU温度
"""


def get_gpu_temp_by_gpu_index(gpu_index):
    gtemp = 0
    try:
        nr = str(gpu_index + 3)
        usagestring = os.popen('nvidia-smi dmon -c 1  |  awk \'NR=='+nr+'{print $3}\'').read()
        gtemp = usagestring[1].split(" ")
    except BaseException as e:
        print(str(e))
    finally:
        return gtemp


"""
    获取当前系统的资源使用情况
"""


def get_sys_resource_usage():
    free_mem, bi, bo, cpu_us, cpu_sys, cpu_id, cpu_wa = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    try:
        usagestring = os.popen('vmstat | awk \'NR==3{print $4,$9,$10,$13,$14,$15,$16}\'').read()
        free_mem, bi, bo, cpu_us, cpu_sys, cpu_id, cpu_wa = usagestring.replace('\n','').split(" ")
    except BaseException as e:
        print(str(e))
    finally:
        return free_mem, bi, bo, cpu_us, cpu_sys, cpu_id, cpu_wa


"""
    执行算法
    入参：demo_name(demo 名称)，run_string(执行参数)，log_file（log文件名称）
    返回值：进程号
"""


def demo_run(demo_name, run_string, log_file):
    pid = -1
    try:
        print("run_string:" + run_string)
        print("log_file:" + log_file)
        os.environ['run_string'] = str(run_string)
        os.environ['log_file'] = str(log_file)
        pid = os.popen('nohup $run_string >> $log_file & echo $!').read()
        pid = pid.replace("\n", "")
        print("demo_run pid is", pid)
    except BaseException as e:
        print(str(e))
    finally:
        return pid


def proc_exist(pid):
    try:
        is_exist = False
        if (str(pid) == os.popen('ps aux | grep ' + str(pid) + ' | awk \' NR==1 {print $2}\'').read().replace('\n','')):
            is_exist = True

    except BaseException as e:
        print(str(e))
    finally:
        return is_exist


"""
    文件保存
"""


def save_json_to_file(json_info, log_file):
    with open(log_file, "w") as json_file:
        json_file.write(json.dumps(json_info))


"""
    获取最大、最小、平均值、25分位、50分位、75分位、95分位
"""


def get_usage_statistics(info_list):
    list_max = 0
    list_min = 0
    list_mean = 0
    the25th = 0
    the50th = 0
    the75th = 0
    the95th = 0

    try:
        list_max = np.max(info_list)
        list_min = np.min(info_list)
        list_mean = np.mean(info_list)
        the50th = np.percentile(info_list, 0.5)
        the25th = np.percentile(info_list, 0.25)
        the75th = np.percentile(info_list, 0.75)
        the95th = np.percentile(info_list, 0.95)
    except BaseException as e:
        print(str(e))
    finally:
        return {"max":list_max, "min":list_min, "mean":list_mean, "the25":the25th, "the50":the50th, "the75":the75th, "the95":the95th}


"""
    获取最大、最小、平均值、25分位、50分位、75分位、95分位
    入参：info_list，param_string（需要获取分位置的参数）, start_time, pid
    返回值：字典
"""


def set_usage_statistics(type, info_list, param_string, start_time, pid):
    usage_dic = {'type': type,'start_time': start_time, "pid": pid, "usage_list": info_list}
    try:
        for i in param_string:
            info_tmp_list = []
            for j in info_list:
                info_tmp_list.append(float(j[i]))
            tmp_info = get_usage_statistics(info_tmp_list)
            usage_dic[i] = tmp_info
    except BaseException as e:
        print(str(e))
    finally:
        return usage_dic

# 执行函数,运行一次
def run_once(demo_name, cur_count, run_string, gpu_num, log_dir, demo_lable):
    # 获取当前时间
    start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    log_file = log_dir + demo_name + '_' + str(start_time) + '_' + str(cur_count)
    pid = demo_run(demo_name, run_string, log_file)

    # CPU 资源使用率
    cpu_pid_usage_list = []
    # GPU 某进程资源使用率
    gpu_pid_usage_list = []
    # GPU 资源使用率
    gpu_index_usage_list = []
    # 系统资源使用率
    sys_usage_list = []

    try:
        while (proc_exist(str(pid))):
            exit_proc = False
            for gpu_index in range(1, gpu_num + 1):
                gtemp = get_gpu_temp_by_gpu_index(gpu_index)
                if (gtemp > 80):
                    exit_proc = True
            if (exit_proc == True):
                # 退出验证
                usagestring = os.popen('kill -9 ' + pid).read()
                return -1  # 退出循环
            cur_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
            pid, cpu_usage, mem_usage = get_cpu_resource_usage_by_pid(pid)
            cpu_usage_by_pid = {'cur_time': cur_time, 'pid': pid, 'cpu_usage': cpu_usage, 'mem_usage': mem_usage}
            cpu_pid_usage_list.append(cpu_usage_by_pid)
            pid, gpu_index, gpu_usage, mem_usage, pid_name = get_gpu_resource_usage_by_pid(pid)
            gpu_usage_by_pid = {'cur_time': cur_time, 'pid': pid, 'gpu_index': gpu_index, 'gpu_usage': gpu_usage,
                                'gpu_mem_usage': mem_usage, 'pid_name': pid_name}
            gpu_pid_usage_list.append(gpu_usage_by_pid)
            for gpu_index in range(1, gpu_num + 1):
                gpu_index, pwr, gtemp, gpu_usage, mem_usage, mclk, pclk = get_gpu_resource_usage_by_gpu_index(
                    gpu_index - 1)
                gpu_usage_by_index = {'cur_time': cur_time, 'gpu_index': gpu_index, 'pwr': pwr, 'gtemp': gtemp,
                                      'gpu_usage': gpu_usage, 'mem_usage': mem_usage, 'mclk': mclk, 'pclk': pclk}
                gpu_index_usage_list.append(gpu_usage_by_index)

            free_mem, bi, bo, cpu_us, cpu_sys, cpu_id, cpu_wa = get_sys_resource_usage()
            sys_usage = {'cur_time': cur_time, 'free_mem': free_mem, 'bi': bi, 'bo': bo, 'cpu_us': cpu_us,
                         'cpu_sys': cpu_sys, 'cpu_id': cpu_id, 'cpu_wa': cpu_wa}
            sys_usage_list.append(sys_usage)
            time.sleep(1)
    except BaseException as e:
        print(str(e))
    finally:
        usagestring = os.popen('kill -9 ' + pid).read()
        end_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        #print("cpu_pid_usage_list:", cpu_pid_usage_list)
        #print("gpu_pid_usage_list:", gpu_pid_usage_list)
        #print("gpu_index_usage_list:", gpu_index_usage_list)
        #print("sys_usage_list:", sys_usage_list)
        # 写文件
        os.environ['log_file'] = str(log_file)
        # 获取最大最小值， 25分位，50分位，75分位

        # 该进程CPU 资源使用情况
        cpu_pid_usage_dic = set_usage_statistics("cpu_pid_usage",cpu_pid_usage_list, ["cpu_usage", "mem_usage"], start_time, pid)
        gpu_pid_usage_dic = set_usage_statistics("gpu_pid_usage",gpu_pid_usage_list, ["gpu_mem_usage", "gpu_usage"], start_time, pid)
        gpu_index_usage_dic = set_usage_statistics("gpu_index_usage",gpu_index_usage_list,
                                                   ['pwr', 'gtemp', 'gpu_usage', 'mem_usage', 'mclk', 'pclk'],
                                                   start_time, pid)
        sys_usage_dic = set_usage_statistics("sys_usage",sys_usage_list,
                                             ['free_mem', 'bi', 'bo', 'cpu_us', 'cpu_sys', 'cpu_id', 'cpu_wa'],
                                             start_time, pid)
        cpu_pid_usage_dic['cur_count'] = cur_count
        gpu_pid_usage_dic['cur_count'] = cur_count
        gpu_index_usage_dic['cur_count'] = cur_count
        sys_usage_dic['cur_count'] = cur_count
        cpu_pid_usage_dic['end_time'] = end_time
        gpu_pid_usage_dic['end_time'] = end_time
        gpu_index_usage_dic['end_time'] = end_time
        sys_usage_dic['end_time'] = end_time
        result_info = {}
        # 写入数据库
        try:
            # 算法运行结果
            demo_result_total_time = 0
            demo_result_total_images = 0
            epoch_train_perplexity =0
            epoch_valid_perplexity = 0
            elapsed_time = 0
            test_perplexity = 0
            with open(log_file, 'r') as f:
                lines = f.readlines()
                # 获取倒数三十行数据
                run_result = lines[-30:]
                # 根据现有结果获取total time:/ total images/sec:

                #reg = re.compile(r"(?<=total time)\d+:")
                #match = reg.search(run_result)
                #print(match.group(0))
                run_result = ''.join(run_result)
                if ("resnet" in demo_name) :
                    try:
                        total_time = re.findall(r"total time:(.+?)\n", run_result)[0]
                        total_time = float(total_time)
                        demo_result_total_time = round(total_time, 2)
                        total_images = re.findall(r"total images/sec:(.+?)\n", run_result)[0]
                        total_images = float(total_images)
                        demo_result_total_images = round(total_images, 2)
                    except BaseException as e:
                        print(str(e))
                elif ("lstm" in demo_name) :
                    try:
                        #epoch_train_perplexity = re.findall(r"Epoch:13 Train Perplexity:(.+?)\n", run_result)[0]
                        #epoch_valid_perplexity = re.findall(r"Epoch:13 Valid Perplexity:(.+?)\n", run_result)[0]
                        elapsed_time = re.findall(r"Elapsed Time : (.+?)\n", run_result)[0]
                        #test_perplexity = re.findall(r"Test Perplexity:(.+?)\n", run_result)[0]
                        if (elapsed_time != None):
                            elapsed_time = float(elapsed_time)
                            #epoch_train_perplexity = float(epoch_train_perplexity)
                            #epoch_valid_perplexity = float(epoch_valid_perplexity)

                            #test_perplexity = float(test_perplexity)
                        #epoch_train_perplexity = round(epoch_train_perplexity, 2)
                        #epoch_valid_perplexity = round(epoch_valid_perplexity, 2)
                            elapsed_time = round(elapsed_time, 2)
                        #test_perplexity = round(test_perplexity, 2)
                    except BaseException as e:
                        print(str(e))

            save_info = {}
            save_info['start_time'] = start_time
            save_info['end_time'] = end_time
            save_info['pid'] = pid
            save_info['demo_name'] = demo_name
            save_info['demo_lable'] = demo_lable
            save_info['cur_count'] = cur_count
            save_info['cpu_pid_usage'] = cpu_pid_usage_dic
            save_info['gpu_pid_usage_dic'] = gpu_pid_usage_dic
            save_info['gpu_index_usage_dic'] = gpu_index_usage_dic
            save_info['sys_usage_dic'] = sys_usage_dic
            save_info['run_result'] = run_result

            db = pymongo.MongoClient("172.24.27.63", 27017)['ai']
            collecttion = db.get_collection("usage_info")
            collecttion.insert(save_info)

            result_info['start_time'] = start_time
            result_info['end_time'] = end_time
            result_info['pid'] = pid
            result_info['demo_name'] = demo_name
            result_info['demo_lable'] = demo_lable
            result_info['cur_count'] = cur_count
            if ("resnet" in demo_name):
                result_info['result_total_time'] = demo_result_total_time
                result_info['result_total_images_per_s'] = demo_result_total_images
            elif ("lstm" in demo_name):
                result_info['epoch_train_perplexity'] = epoch_train_perplexity
                result_info['epoch_valid_perplexity'] = epoch_valid_perplexity
                result_info['elapsed_time'] = elapsed_time
                result_info['test_perplexity'] = test_perplexity
            result_info['gpu_min'] = gpu_pid_usage_dic['gpu_usage']['min']
            result_info['gpu_max'] = gpu_pid_usage_dic['gpu_usage']['max']
            result_info['gpu_mean'] = gpu_pid_usage_dic['gpu_usage']['mean']
            result_info['gpu_mem_min'] = gpu_pid_usage_dic['gpu_mem_usage']['min']
            result_info['gpu_mem_max'] = gpu_pid_usage_dic['gpu_mem_usage']['max']
            result_info['gpu_mem_mean'] = gpu_pid_usage_dic['gpu_mem_usage']['mean']

            result_info['mem_min'] = cpu_pid_usage_dic['mem_usage']['min']
            result_info['mem_max'] = cpu_pid_usage_dic['mem_usage']['max']
            result_info['mem_mean'] = cpu_pid_usage_dic['mem_usage']['mean']
            result_info['cpu_min'] = cpu_pid_usage_dic['cpu_usage']['min']
            result_info['cpu_max'] = cpu_pid_usage_dic['cpu_usage']['max']
            result_info['cpu_mean'] = cpu_pid_usage_dic['cpu_usage']['mean']

            result_info['gpu_pwr_min'] = gpu_index_usage_dic['pwr']['min']
            result_info['gpu_pwr_max'] = gpu_index_usage_dic['pwr']['max']
            result_info['gpu_pwr_mean'] = gpu_index_usage_dic['pwr']['mean']
            result_info['gpu_temp_min'] = gpu_index_usage_dic['gtemp']['min']
            result_info['gpu_temp_max'] = gpu_index_usage_dic['gtemp']['max']
            result_info['gpu_temp_mean'] = gpu_index_usage_dic['gtemp']['mean']
            result_info['gpu_total_usage_min'] = gpu_index_usage_dic['gpu_usage']['min']
            result_info['gpu_total_usage_max'] = gpu_index_usage_dic['gpu_usage']['max']
            result_info['gpu_total_usage_mean'] = gpu_index_usage_dic['gpu_usage']['mean']
            result_info['gpu_total_mem_min'] = gpu_index_usage_dic['mem_usage']['min']
            result_info['gpu_total_mem_max'] = gpu_index_usage_dic['mem_usage']['max']
            result_info['gpu_total_mem_mean'] = gpu_index_usage_dic['mem_usage']['mean']
            result_info['gpu_mclk_min'] = gpu_index_usage_dic['mclk']['min']
            result_info['gpu_mclk_max'] = gpu_index_usage_dic['mclk']['max']
            result_info['gpu_mclk_mean'] = gpu_index_usage_dic['mclk']['mean']
            result_info['gpu_pclk_min'] = gpu_index_usage_dic['pclk']['min']
            result_info['gpu_pclk_max'] = gpu_index_usage_dic['pclk']['max']
            result_info['gpu_pclk_mean'] = gpu_index_usage_dic['pclk']['mean']

            result_info['cpu_wa_min'] = sys_usage_dic['cpu_wa']['mean']
            result_info['cpu_wa_max'] = sys_usage_dic['cpu_wa']['mean']
            result_info['cpu_wa_mean'] = sys_usage_dic['cpu_wa']['mean']

            print("result info :", result_info)

            collecttion_result = db.get_collection("usage_result_info")
            collecttion_result.insert(result_info)
        except BaseException as e:
            print(str(e))
        finally:
            cpu_pid_usage_dic = json.dumps(cpu_pid_usage_dic)
            gpu_pid_usage_dic = json.dumps(gpu_pid_usage_dic)
            gpu_index_usage_dic = json.dumps(gpu_index_usage_dic)
            sys_usage_dic = json.dumps(sys_usage_dic)
            #print("cpu_pid_usage_dic :", cpu_pid_usage_dic)
            #print("gpu_pid_usage_dic :", gpu_pid_usage_dic)
            #print("gpu_index_usage_dic :", gpu_index_usage_dic)
            #print("sys_usage_dic :", sys_usage_dic)
            save_json_to_file(cpu_pid_usage_dic, log_file + '_cpu_pid_usage_dic')
            save_json_to_file(gpu_pid_usage_dic, log_file + '_gpu_pid_usage_dic')
            save_json_to_file(gpu_index_usage_dic, log_file + '_gpu_index_usage_dic')
            save_json_to_file(sys_usage_dic, log_file + '_sys_usage_dic')
            return result_info

# 运行多次
def run_much(demo_name, run_string, gpu_num, log_dir, run_count):
    result_info_list = []
    cur_count = 1
    demo_lable = demo_name+time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    for i in range(1, run_count + 1):
        # 判断GPU 温度是否过高
        exit_proc = False
        for gpu_index in range(1, gpu_num + 1):
            gtemp = get_gpu_temp_by_gpu_index(gpu_index)
            if (gtemp > 60):
                exit_proc = True
        if (exit_proc == True):
            continue

        result_info = run_once(demo_name, cur_count, run_string, gpu_num, log_dir, demo_lable)
        if (result_info == -1) :
            continue
        result_info_list.append(result_info)
        cur_count += 1
    print(result_info_list)

    # 计算平均值

if __name__ == '__main__':
    run_resnet50 = False
    run_lstm = True

    if (run_resnet50) :
        demo_name = "resnet50"
        cur_count = 1
        run_string = "python3.6 /data/lxw/benchmarks-master/benchmarks/scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py " \
                     "--num_gpus=1 --batch_size=32 --model=resnet50  --variable_update=parameter_server"
        gpu_num = 1
        log_dir = "/home/czc/test/"
        run_count = 3
        #run_once(demo_name, cur_count, run_string, gpu_num, log_dir)
        run_much(demo_name, run_string, gpu_num, log_dir, run_count)
    if (run_lstm):
        demo_name = "lstm_ptb"
        cur_count = 1
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        run_string = "python3.6 /data/lxw/ptb/ptb_word_lm.py --data_path=/data/lxw/simple-examples/data/ " \
                                             "--model=medium --use_fp16=True --rnn_mode=CUDNN --num_gpus=1"
        gpu_num = 1
        log_dir = "/home/czc/test/"
        run_count = 1
        #run_once(demo_name, cur_count, run_string, gpu_num, log_dir)
        run_much(demo_name, run_string, gpu_num, log_dir, run_count)

    pass
