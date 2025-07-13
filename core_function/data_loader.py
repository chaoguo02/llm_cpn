import pandas as pd
import numpy as np
from entity.Node import Node
from entity.Task import Task
from entity.Taskflow import TaskFlow


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

def GoogleClusterReader(Filename,num):
    data = pd.read_csv(Filename, usecols=['plan_cpu', 'plan_mem'], nrows=num)
    data.dropna(subset=['plan_cpu', 'plan_mem'], inplace=True)#去掉空值的行数
    data = data[(data['plan_cp'] != 0) & (data['plan_mem'] != 0)]  # 去掉为0的值，否则时间为null
    data['plan_cpu'] = data['plan_cpu'] * 400 * 2 # 600个核 2GHz
    data['plan_mem'] = data['plan_mem'] * 512  # 512GB 内存
    # 将数据转换为 NumPy 数组
    data_array = data.to_numpy()
    return data_array

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

def createTask(dataSet,taskflow_id,all_arrive_time):#用于动态工作流
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
        task = Task(id, taskflow_id,realruntime,cpu,ram,  gpu, all_arrive_time)
        taskList.append(task)
        id += 1
    return taskList

def createNode(node_num):
    proportions = [0.6, 0.3, 0.1]
    node_types = ["EdgeNode", "FogNode", "CloudNode"]
    gpu_process     = [10000,   100000,   1000000]
    cpu_process      = [10000,   100000,   500000]
    cpu_capacities  = [100,     100,      100]
    gpu_capacities  = [100,     100,      100]
    ram_capacities  = [100,     250,      512]
    prices          = [0.0000029, 0.0000229, 0.000115]
    bandwidths      = [100,     100,      100]
    delays          = [100,     100,      100]
    node_counts = [int(p * node_num) for p in proportions]
    nodes = []
    node_id = 0
    for i in range(len(node_types)):
        for _ in range(node_counts[i]):
            node = Node(
                id=node_id,
                gpu_process=gpu_process[i],
                cpu_process=cpu_process[i],
                bandwidth=bandwidths[i],
                delay=delays[i],
                cpu_capacity=cpu_capacities[i],
                ram_capacity=ram_capacities[i],
                gpu_capacity=gpu_capacities[i],
                node_type=node_types[i],
                price=prices[i]
            )
            nodes.append(node)
            node_id += 1

    return nodes

def createTaskFlows(num_TaskFlow,genre,seed): #创建多个工作流 #k为任务流随机种子
    lambda_rate = 1  # 平均到达速率 (λ，每单位时间平均到达任务数)
    np.random.seed(seed)
    interarrival_times = np.random.exponential(1 / lambda_rate, num_TaskFlow)  # 会生成一个包含 num_tasks 个任务到达时间间隔的数组

    # 计算每个任务的到达时间（通过累加间隔时间）
    arrival_times = np.cumsum(interarrival_times)
    taskflows=[TaskFlow(id,arrival_time,genre,id+(seed)*num_TaskFlow) for id, arrival_time in zip(range(num_TaskFlow), arrival_times)]
    return taskflows
