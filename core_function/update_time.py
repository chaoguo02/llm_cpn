import numpy as np

import core_function.config as fig

def present_time_update(present_time, taskflows):
    # 只更新那些未完成任务的当前时间，所以当一个任务已完成时，它的当前时间就是完成时间
    for taskflow in taskflows:
        for task in taskflow.tasks:
            if task.finish is False:
                task.present_time = present_time

def update_next_free_time(node, current_time):
    """重新计算 next_free_time：包括当前任务执行完后的等待队列总时间"""
    simulated_time = max(current_time, node.begin_idle_time)  # 当前任务执行结束时间
    for task in node.waiting_queue:
        task_exec = computing_Task(task, node)
        task_upload = computing_upload_time(task, node)
        simulated_time += task_exec + task_upload
    node.next_free_time = simulated_time

#
def predict_node_idle_time(node):
    """
    计算节点的预期空闲时间，考虑 waiting_queue 中所有任务的执行时间
    """
    predicted_time = node.begin_idle_time
    for queued_task in node.waiting_queue:
        predicted_time += computing_Task(queued_task, node) + computing_upload_time(queued_task, node)
    return predicted_time


def estimated_finish_time(task,node):
    compute_time = computing_Task(task, node)
    upload_time = computing_upload_time(task, node)
    start_time = max(task.present_time, node.begin_idle_time)
    return start_time + compute_time + upload_time

def find_earlist_time(queue1,queue2):
    # 寻找下一个操作时
    queue1.sort(key=lambda x: x[1])
    queue2.sort(key=lambda x: x[1])
    task_queue1 = []
    task_queue2 = []
    if len(queue1)!=0 and len(queue2)!=0:
        min_time=min(queue1[0][1],queue2[0][1])
        for event in queue1:
            if event[1]==min_time:
                task_queue1.append(event[0])
            elif event[1] > min_time:
                break
        for event in queue2:
            if event[1]==min_time:
                task_queue2.append(event[0])
            elif event[1] > min_time:
                break
    elif len(queue1)==0  and len(queue2)!=0:
        min_time = queue2[0][1]
        for event in queue2:
            if event[1]==min_time:
                task_queue2.append(event[0])
            elif event[1] > min_time:
                break
    elif len(queue1)!=0  and len(queue2)==0:
        min_time = queue1[0][1]
        for event in queue1:
            if event[1]==min_time:
                task_queue1.append(event[0])
            elif event[1] > min_time:
                break
    return queue1,queue2,task_queue1,task_queue2

def get_fluctuation(node):
    # 默认值
    base_loc = 1.0

    if node.type == 'CloudNode':
        scale = 0.03
    elif node.type == 'FogNode':
        scale = 0.06
    elif node.type == 'EdgeNode':
        scale = 0.10
    else:
        scale = 0.05  # 未知类型默认中等波动

    fluctuation = np.random.normal(loc=base_loc, scale=scale)
    return max(0.7, fluctuation)

# computing_Task计算的是 指定的task在该node上的执行时间
def computing_Task(task,node):
    task_value=[task.runtime * fig.Average_MIPS * task.cpu_require,task.runtime *fig.Average_TFLOPS*task.gpu_require]
    # f_fluctuation = get_fluctuation(node)
    return task_value[0] / (task.cpu_require * node.cpu_process)  #task.cpu_require要求的核数，node.cpu_process单个核的计算能力


# computing_upload_time计算的是 将task上传到node所需的时间
def computing_upload_time(task,node):#设计算机为64位  双精度浮点数 1MIPS=8MB/S
    # 根据不同的node类型，添加base_upload_time 50,300,400
    base_upload = 0
    if node.node_type == 'EdgeNode':
        base_upload = 50
    elif node.node_type == 'FogNode':
        base_upload = 300
    elif node.node_type == 'CloudNode':
        base_upload = 400
    task.want_get_bandwidth(node)
    exetime = 8 * task.runtime * fig.Average_MIPS * task.cpu_require / (task.get_bandwidth * 1000000)  # s 只考虑cpu
    return exetime + base_upload

def computing_upload_cost(task,node):#0.29元/MB 中国移动
    task.want_get_bandwidth(node)
    cost=8 * task.runtime * fig.Average_MIPS * task.cpu_require*0.29 *1/7  #$
    return cost
