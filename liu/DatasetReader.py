import pandas as pd
from sklearn.model_selection import train_test_split
# from Class.Task2 import Task
from Class.Task3 import Task
# from Class.Node2 import Node
from Class.Node3 import Node
import random
import numpy as np
from . import ConstantConfig as fig

"""
Alibaba 数据集读取
"""
"""
def AlibabaClusterReader(Filename,num):  顺序找num个
    data = pd.read_csv(Filename, usecols=['start_time','end_time', 'plan_cpu', 'plan_mem', 'plan_gpu'], nrows=num)
    data.dropna(subset=['start_time','end_time', 'plan_cpu', 'plan_mem', 'plan_gpu'], inplace=True)#去掉空值的行数
    data = data[ (data['start_time'] != None) & (data['end_time'] != None) & (data['plan_cpu'] != 0) & (data['plan_mem'] != 0) &
                (data['plan_gpu'] != 0)] #去掉为0的值，否则时间为null
    data['plan_cpu'] = data['plan_cpu'] / 100 #得到核数
    # 将数据转换为 NumPy 数组
    data_array = data.to_numpy()
    return data_array
"""






def AlibabaClusterReader(Filename, num, start_percentage, end_percentage, seed):
    # 设置随机种子


    # 读取数据集，并且指定所需的列
    data = pd.read_csv(Filename, usecols=['start_time', 'end_time', 'plan_cpu', 'plan_mem', 'plan_gpu'])

    # 删除包含空值的行
    data.dropna(subset=['start_time', 'end_time', 'plan_cpu', 'plan_mem', 'plan_gpu'], inplace=True)

    # 删除 plan_cpu, plan_mem, plan_gpu 中为 0 的行
    data = data[(data['start_time'] != None) & (data['end_time'] != None) &
                (data['plan_cpu'] != 0) & (data['plan_mem'] != 0) &
                (data['plan_gpu'] != 0)]

    # 获取数据集的总行数
    num_rows = len(data)

    # 计算前 start_percentage 到 end_percentage 的行索引范围
    start_row = int(num_rows * start_percentage / 100)
    end_row = int(num_rows * end_percentage / 100)

    # 选择从 start_row 到 end_row 之间的行
    selected_data = data.iloc[start_row:end_row]

    # 从选定的部分数据中随机抽取 num 行数据
    sampled_data = selected_data.sample(n=num, random_state=seed)

    # 将 plan_cpu 转换为核数
    sampled_data['plan_cpu'] = sampled_data['plan_cpu'] / 100
    sampled_data['plan_gpu'] = sampled_data['plan_gpu'] / 100
    # 将数据转换为 NumPy 数组
    data_array = sampled_data.to_numpy()

    return data_array







"""
Google配置(2011)

CPU： 双路 Intel Xeon 处理器，每个处理器 6 个核心，总共 12 个核心，主频约 2.0 GHz。
内存（RAM）： 64 GB DDR3 内存。
存储（DISK）： 2 TB（2000 GB） 的 HDD 硬盘存储。

"""

def GoogleClusterReader(Filename,num):
    data = pd.read_csv(Filename, usecols=['plan_cpu', 'plan_mem'], nrows=num)
    data.dropna(subset=['plan_cpu', 'plan_mem'], inplace=True)#去掉空值的行数
    data = data[(data['plan_cp'] != 0) & (data['plan_mem'] != 0)]  # 去掉为0的值，否则时间为null
    data['plan_cpu'] = data['plan_cpu'] * 400 * 2 # 600个核 2GHz
    data['plan_mem'] = data['plan_mem'] * 512  # 512GB 内存
    # 将数据转换为 NumPy 数组
    data_array = data.to_numpy()
    return data_array

"""
测试集与训练集的划分
"""


# 生成模拟数据
def generate_data(taskList, nodeList):
    data = []
    for task in taskList:
        # 随机生成任务的 CPU 和 RAM 需求
        task_cpu = task.cpu_capacity
        task_ram = task.ram_capacity

        for node in nodeList:
            # 随机生成节点的 CPU 和 RAM 容量
            node_cpu = node.cpu_capacity
            node_ram = node.ram_capacity

            # 判定任务是否能够调度到该节点
            can_schedule = 1 if task_cpu <= node_cpu and task_ram <= node_ram else 0

            # 添加任务和节点组合到数据列表中
            data.append([task_cpu, task_ram,node_cpu, node_ram, can_schedule])

    # 转换为 DataFrame
    df = pd.DataFrame(data,
                      columns=['Task_CPU', 'Task_RAM','Node_CPU', 'Node_RAM', 'Can_Schedule'])
    return df

# def createTask(dataSet):
#     taskList = []
#     id = 0
#     for datarow in dataSet:
#         given_starttime = float(datarow[0])
#         given_endtime = float(datarow[1])
#         cpu = datarow[2]
#         ram = datarow[3]
#         gpu = datarow[4]
#         realruntime = given_endtime - given_starttime
#         #print(cpu,ram)
#         task = Task(id, runtime=realruntime,cpu_require=cpu,ram_require=ram, gpu_require = gpu)
#         taskList.append(task)
#         id += 1
#     return taskList



def createTask(dataSet,self_id,all_arrive_time):#用于动态工作流
    taskList = []
    id = 0
    for datarow in dataSet:
        given_starttime = float(datarow[0])
        given_endtime = float(datarow[1])
        cpu = datarow[2]
        ram = datarow[3]
        gpu = datarow[4]
        realruntime = given_endtime - given_starttime
        #print(cpu,ram)
        task = Task(id, self_id,realruntime,cpu,ram,  gpu, all_arrive_time)
        taskList.append(task)
        id += 1
    return taskList


# def createNode(nodeNum):#用于静态
#     nodes = []
#     for i in range(nodeNum):
#         # 随机选择节点类型
#         node_type = random.choice(["CloudNode", "EdgeNode", "FogNode"])
#         #cpu_process = fig.Ryzen_9_5950X_CORE_MIPS
#         cpu_process = random.choice([fig.Ryzen_9_5950X_CORE_MIPS, fig.Ryzen_7_5800X_CORE_MIPS,fig.Ryzen_5_5600X_CORE_MIPS, fig.Ryzen_5_7600X_CORE_MIPS])
#         gpu_process = random.choice([fig.NVIDIA_RTX_4090_TFLOPS, fig.NVIDIA_RTX_4090_TFLOPS,fig.NVIDIA_A100_TFLOPS, fig.NVIDIA_V100_TFLOPS])
#
#         #gpu_process = fig.NVIDIA_RTX_4090_TFLOPS
#         cpu_capacity = 100 #初始化
#         ram_capacity = 100#初始化
#         gpu_capacity = 300
#         bandwidth = 100
#         delay = 100
#         # 随机选择 processing_capacity 和其他资源范围
#         if node_type == "CloudNode":
#             cpu_capacity = random.randint(25, 40)
#             ram_capacity = random.randint(120, 160)
#             bandwidth = random.randint(10, 20)  # MHZ
#             delay = random.randint(50, 80)  # ms
#         elif node_type == "EdgeNode":
#             cpu_capacity = random.randint(20, 35)
#             ram_capacity = random.randint(80, 120)
#             bandwidth = random.randint(2, 5)  # MHZ
#             delay = random.randint(10, 30)  # ms
#         elif node_type == "FogNode":
#             cpu_capacity = random.randint(15, 30)
#             ram_capacity = random.randint(50, 80)
#             bandwidth = random.randint(5, 10)  # MHZ
#             delay = random.randint(30, 50)  # ms
#
#         # 创建节点并添加到节点列表中
#         node = Node(i,cpu_process,gpu_process,bandwidth,delay, cpu_capacity, ram_capacity, gpu_capacity, node_type)
#         nodes.append(node)
#
#     return nodes

def createNode(nodeNum,rate): #genre用于表示不同规模的网络，rate表示云边端不同的比例
    rate_value=[[0.2,0.4,0.4],[0.2,0.4,0.4],[0.3,0.3,0.4]]
    gpu_computing_power=[1000000,100000,10000]
    cpu_computing_power=[500000,100000,10000]
    nodes = []
    node_number=  [int(x*nodeNum) for x in rate_value[rate]]
    Num=0
    for i in range(len(node_number)):
        for j in range(node_number[i]):
            cpu_capacity = 100  # 初始化

            gpu_capacity = 100
            bandwidth = 100
            delay = 100
            if i==0:
                node_type="CloudNode"
                price=0.000115 #单价每秒$
                ram_capacity = 512 # 初始化
            if i==1:
                node_type="FogNode"
                price =0.0000229
                ram_capacity = 250  # 初始化
            if i==2:
                node_type="EdgeNode"
                price =0.0000029
                ram_capacity = 100  # 初始化
            # 创建节点并添加到节点列表中
            node = Node(Num, gpu_computing_power[i], cpu_computing_power[i], bandwidth, delay, cpu_capacity, ram_capacity, gpu_capacity,
                            node_type, price)
            nodes.append(node)
            Num+=1

    return nodes


