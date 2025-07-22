import numpy as np
import core_function.config as fig

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
    print(fluctuation)
    return max(0.7, fluctuation)

# computing_Task计算的是 指定的task在该node上的执行时间
def computing_Task(task,node):
    task_value=[task.runtime * fig.Average_MIPS * task.cpu_require,task.runtime *fig.Average_TFLOPS*task.gpu_require]
    f_fluctuation = get_fluctuation(node)

    return task_value[0] / (task.cpu_require * node.cpu_process) * f_fluctuation #task.cpu_require要求的核数，node.cpu_process单个核的计算能力
