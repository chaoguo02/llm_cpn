
import numpy as np
from deap import gp
from ..liu.exetime import computing_Task, computing_upload_time#(task,node) #..为返回上次导入


def present_time_update(present_time, taskflows):  # 只更新那些未完成任务的当前时间，所以当一个任务已完成时，它的当前时间就是完成时间
    for taskflow in taskflows:
        for task in taskflow.tasks:
            if task.finish is False:
                task.present_time = present_time


def find_earlist_time(queue1, queue2):
    # 使用 sort() 方法排序
    queue1.sort(key=lambda x: x[1])
    queue2.sort(key=lambda x: x[1])
    task_queue1 = []
    task_queue2 = []
    if len(queue1) != 0 and len(queue2) != 0:
        min_time = min(queue1[0][1], queue2[0][1])
        for event in queue1:
            if event[1] == min_time:
                task_queue1.append(event[0])
        for event in queue2:
            if event[1] == min_time:
                task_queue2.append(event[0])
    elif len(queue1) == 0 and len(queue2) != 0:
        min_time = queue2[0][1]
        for event in queue2:
            if event[1] == min_time:
                task_queue2.append(event[0])
    elif len(queue1) != 0 and len(queue2) == 0:
        min_time = queue1[0][1]
        for event in queue1:
            if event[1] == min_time:
                task_queue1.append(event[0])
    return queue1, queue2, task_queue1, task_queue2


def get_score(item):
    return item[1]


def decode1(individual, task, nodes, taskflows, pset):  # decode1函数中需要增加一个约束条件，判断节点的ram，gpu，cpu够不够用，要在够用的节点中选择
    heuristic_1 = gp.compile(expr=individual[0], pset=pset[0])
    scores = []
    k = task.taskflow_id  # 任务所在任务流，在多任务流中的位置
    for node in nodes:
        heuristic_score = heuristic_1(computing_Task(task, node), task.present_time - task.arrivetime,
                                      len(task.descendant), computing_upload_time(task, node))
        scores.append((node, heuristic_score))
    i = max(scores, key=get_score)[0]
    """

    为什么scores有时会为空
    """

    # print(individual[0],i[0])
    return i  # 找到节点本身


def get_task_score(item):
    return item[1]


def decode2(individual, node, taskflows, nodes, pset):
    heuristic_2 = gp.compile(expr=individual[1], pset=pset[1])
    scores = []
    for task in node.waiting_queue:
        k = task.taskflow_id  # 任务所在任务流，在多任务流中的位置
        heuristic_score = heuristic_2(computing_Task(task, node), task.present_time - task.arrivetime,
                                      len(task.descendant),
                                      taskflows[k].find_descendant_avg_time(taskflows, task, nodes),
                                      0.1 * computing_upload_time(task, node))  # 任务给后继传递消息的时间取上传时间的0.1
        scores.append((task, heuristic_score))
    # print(scores)
    # i= max(scores, key=lambda c: c[1])[0]
    i = max(scores, key=get_task_score)[0]
    return i  # 返回找到的任务本身


def work_processing(individual, taskflows, nodes, pset):  # taskflows本体和附属属性有没有改变？※等号传的都是地址
    queue1 = []  # 未执行任务队列
    queue2 = []  # 正在执行的任务队列
    present_time = 0
    present_time_update(present_time, taskflows)

    """
    queue中元素为元组 （任务本身，任务的开始时间或任务的结束时间）
    """
    for taskflow in taskflows:
        tasks = taskflow.find_predecessor_is_zero()
        for task in tasks:
            queue1.append((task, task.arrivetime))  # 元组
    while (len(queue2) != 0 or len(queue1) != 0):
        queue1, queue2, task_queue1, task_queue2 = find_earlist_time(queue1,
                                                                     queue2)  # task_queue1找到所有未执行任务队列中最早的事件，并为queue按照时间排序

        if len(task_queue1) != 0:  # 在queue1和queue2中时间最小的任务在queue1
            present_time = queue1[0][1]
            present_time_update(present_time,
                                taskflows=taskflows)  # 更新所有queue2和queue1中任务和所有node中等待队列中任务的当前时间 和预处理任务流中所有没上queue1的任务当前时间

            for task in task_queue1:
                node = decode1(individual, task, nodes, taskflows,
                               pset)  # 为所有满足要求的任务选择节点,decode1函数中需要增加一个约束条件，判断节点的ram，gpu，cpu够不够用，要在够用的节点中选择
                node.completed_tasks_number += 1
                task.node = node
                if task.present_time >= node.begin_idle_time:  # task的当前时间大于等于节点的空闲时间
                    computing_Task_time = computing_Task(task, node) + computing_upload_time(task, node)

                    endtime = task.present_time + computing_Task_time  # task完成时间=task当前时间+task计算时间 round(a, 1)
                    task.endtime = endtime
                    queue2.append((task, endtime))  # 添加元组
                    node.begin_idle_time = endtime  # node的空闲时间修改为task完成时间
                    node.completed_runningtime += computing_Task(task, node)  # 添加节点已经工作了多少时间
                else:
                    node.waiting_queue.append(task)
                queue1 = [item for item in queue1 if item[0] != task]  # 在queue1中删除任务

        if len(task_queue2) != 0:  # 在queue1和queue2中时间最小的任务在queue2
            for finish_event in task_queue2:  # 标记任务已完成
                finish_event.finish = True
            present_time = queue2[0][1]
            present_time_update(present_time,
                                taskflows=taskflows)  # 更新所有queue2和queue1中任务和所有node中等待队列中任务的当前时间 和预处理任务流中所有没上queue1的任务当前时间(除了已完成的任务)

            for task in task_queue2:
                i = task.taskflow_id  # 位置
                TaskFlow = taskflows[i]  # 需要找到taskflow在taskflows中的位置
                if len(task.descendant) != 0:  # 这个任务有后继节点
                    a = task.descendant  # task的所有后继下标
                    id = TaskFlow.tasks.index(task)  # task在tasks中的下标
                    descendant_tasks = []  # 在task的后继中找到所有前置为0的task
                    for b in a:
                        TaskFlow.tasks[b].predecessor.remove(id)  # 所有任务删除这个前继，向后继传递信息
                        if len(TaskFlow.tasks[b].predecessor) == 0:  # 后继任务没有了所有的前继任务
                            descendant_tasks.append(TaskFlow.tasks[b])
                    for descendant_task in descendant_tasks:
                        queue1.append((descendant_task, descendant_task.present_time))  # 添加元组#该任务上queue1,带着当前时间组成的元组

                else:  # 没有后继
                    TaskFlow.finish_time = max(TaskFlow.finish_time, queue2[0][1])  # 更新任务流的完成时间 取最大值
                if len(task.node.waiting_queue) != 0:  # 任务所在节点的等待队列不为空
                    next_task = decode2(individual, task.node, taskflows, nodes, pset)  # 任务所在节点在其所在队列中选择一个任务
                    task.node.waiting_queue.remove(next_task)
                    computing_Task_time = computing_Task(next_task, task.node) + computing_upload_time(next_task,
                                                                                                       task.node) + 0.1 * computing_upload_time(
                        task, task.node)  # 包括上个任务给后继传递消息的时间

                    task.node.completed_runningtime += computing_Task(next_task, task.node)  # 添加节点已经工作了多少时间
                    next_task.endtime = next_task.present_time + computing_Task_time  # task完成时间=task当前时间+task计算时间
                    queue2.append((next_task, next_task.endtime))  # 添加元组
                    task.node.begin_idle_time = next_task.endtime  # node的空闲时间修改为task完成时间 ,这里需要考虑是否修改了node的本体值

                queue2 = [item for item in queue2 if item[0] != task]  # 在queue1中删除任务#在queue2中删除任务

    node_number_data = []  # 负载均衡
    resource_utilization = []  # 资源利用率
    every_node_sumprice = []  # 每个node的花销

    for node in nodes:
        node_number_data.append(node.completed_tasks_number)
        resource_utilization.append(node.completed_runningtime / present_time)
        every_node_sumprice.append(node.completed_runningtime * node.price)

    sumprice = sum(every_node_sumprice)  # 一组工作流的总花销
    std_dev = np.std(node_number_data)
    avg_ru = sum(resource_utilization) / len(resource_utilization)
    sum_time = 0
    taskflows_number = 0
    for j in taskflows[10:]:  # 遍历非前10个任务流算出任务流的平均完成时间
        taskflows_number += 1
        sum_time += (j.finish_time - j.all_arrive_time)
    # return (sum_time/taskflows_number,std_dev)
    # return (sum_time/taskflows_number,avg_ru)
    # return (avg_ru,std_dev)
    # print(sum_time/taskflows_number,sumprice)
    return (sum_time / taskflows_number, sumprice)

