import numpy as np

from entity.Taskflow import TaskFlow


def createTaskFlows(num_TaskFlow,genre,seed): #创建多个工作流 #k为任务流随机种子
    lambda_rate = 1  # 平均到达速率 (λ，每单位时间平均到达任务数)
    np.random.seed(seed)
    interarrival_times = np.random.exponential(1 / lambda_rate, num_TaskFlow)  # 会生成一个包含 num_tasks 个任务到达时间间隔的数组

    # 计算每个任务的到达时间（通过累加间隔时间）
    arrival_times = np.cumsum(interarrival_times)
    taskflows=[TaskFlow(id,arrival_time,genre,id+(seed)*num_TaskFlow) for id, arrival_time in zip(range(num_TaskFlow), arrival_times)]
    for taskflow in taskflows:
        for task in taskflow.tasks:
            # print(task)
            pass
        print(taskflow)
    return taskflows


createTaskFlows(20,0,0)