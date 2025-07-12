import random
from . import ConstantConfig as fig
"""
单个节点总共执行时间由三部分组成
上传时间
下载时间
执行时间
"""
"""
def computing_Task(task,node):
    exeproportion = 0.5 #暂时设置
    cputime = task.runtime * exeproportion   #秒
    #cpu计算时间
    cpulength = cputime * fig.Average_MIPS#（intel和amd的平均值）

    cpuexetime = cpulength / (task.cpu_require * node.cpu_process)  #task.cpu_require要求的核数，node.cpu_process单个核的计算能力

    #gpu计算时间

    gputime = task.runtime * (1- exeproportion)
    gpulength = gputime * fig.Average_FLOPS
    gpuexetime = gpulength / (task.gpu_require * node.gpu_process)

    operation = random.choice(["add", "max"])  # Randomly choose an operation
    exetime = cpuexetime+gpuexetime #

    if operation == "add":
        exetime = cpuexetime + gpuexetime
    else:

        exetime = max(cpuexetime, gpuexetime)

    #exetime = task.cpu_capacity/node.cpu_capacity #任务量除以计算能力
    #exetime = node.cpu_capacity / task.cpu_capacity  # 任务量除以计算能力
    #print("时间，node,task",exetime,node.cpu_capacity,task.cpu_capacity)
    return exetime/1000
"""
def computing_Task(task,node):
    task_value=[task.runtime * fig.Average_MIPS * task.cpu_require,task.runtime *fig.Average_TFLOPS*task.gpu_require]

    cpuexetime = task_value[0] / (task.cpu_require * node.cpu_process)  #task.cpu_require要求的核数，node.cpu_process单个核的计算能力


    #gpuexetime = task_value[1] / (task.gpu_require * node.gpu_process)

    operation = random.choice(["add", "max"])  # Randomly choose an operation
    #exetime = max(cpuexetime,gpuexetime)
    exetime = cpuexetime
    """
    if operation == "add":
        exetime = cpuexetime + gpuexetime
    else:

        exetime = max(cpuexetime, gpuexetime)
    """
    #exetime = task.cpu_capacity/node.cpu_capacity #任务量除以计算能力
    #exetime = node.cpu_capacity / task.cpu_capacity  # 任务量除以计算能力
    #print("时间，node,task",exetime,node.cpu_capacity,task.cpu_capacity)
    return exetime







def computing_upload_time(task,node):#设计算机为64位  双精度浮点数 1MIPS=8MB/S
    task.want_get_bandwidth(node)
    #exetime=8*max(task.runtime * fig.Average_MIPS *task.cpu_require ,task.runtime * fig.Average_TFLOPS*task.gpu_require) / (task.get_bandwidth *1000000)  #s
    exetime = 8 * task.runtime * fig.Average_MIPS * task.cpu_require / (task.get_bandwidth * 1000000)  # s 只考虑cpu
    return exetime





def computing_upload_cost(task,node):#0.29元/MB 中国移动
    task.want_get_bandwidth(node)
    cost=8 * task.runtime * fig.Average_MIPS * task.cpu_require*0.29 *1/7  #$
    return cost



















def isTaskAllocation1(task,node):#考虑capacity
    #print("node,task",node,task)
    if(task.cpu_require <= node.cpu_capacity ):
        if(task.ram_require <= node.ram_capacity):
            if (task.gpu_require <= node.gpu_capacity):
                  return 1
    return 0
def isTaskAllocation2(task,node):#考虑available
    #print("node,task",node,task)
    if(task.cpu_require <= node.cpu_available ):
        if(task.ram_require <= node.ram_available):
            if (task.gpu_require <= node.gpu_available):
                  return 1
    return 0

def upload_Task(tasks,nodes):
    nodes.sort(key=lambda x: x.cpu_capacity, reverse=True)
    for node in nodes:
        if node.assign_task(tasks):
            return True
    return False

def download_Task(tasks,nodes):
    nodes.sort(key=lambda x: x.cpu_capacity, reverse=True)
    for node in nodes:
        if node.assign_task(tasks):
            return True
    return False