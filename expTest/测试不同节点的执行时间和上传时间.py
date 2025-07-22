import core_function.config as fig
from core_function.data_loader import AlibabaClusterReader, createTask, createNode


def computing_Task(task,node):
    task_value=[task.runtime * fig.Average_MIPS * task.cpu_require,task.runtime *fig.Average_TFLOPS*task.gpu_require]
    return task_value[0] / (task.cpu_require * node.cpu_process)  #task.cpu_require要求的核数，node.cpu_process单个核的计算能力


# computing_upload_time计算的是 将task上传到node所需的时间
def computing_upload_time(task,node):#设计算机为64位  双精度浮点数 1MIPS=8MB/S
    base_upload = 0
    # if node.node_type == 'EdgeNode':
    #     base_upload = 50
    # elif node.node_type == 'FogNode':
    #     base_upload = 300
    # elif node.node_type == 'CloudNode':
    #     base_upload = 400
    task.want_get_bandwidth(node)
    exetime = 8 * task.runtime * fig.Average_MIPS * task.cpu_require / (task.get_bandwidth * 1000000)  # s 只考虑cpu
    return exetime + base_upload

Filename = "../Dataset/Alibaba/AlibabaCluster1.csv"
dataSet = AlibabaClusterReader(Filename, 20, 0, 70, 0)  # 在前百分之70中随机找num_tasks个数据
tasks = createTask(dataSet,0,0)

nodes = createNode(10)

print("\n📊 每个任务在每个节点上的执行与上传时间：")
for task in tasks:
    for node in (nodes[0], nodes[6],nodes[9]):
        exec_time = computing_Task(task, node)
        upload_time = computing_upload_time(task, node)
        total = exec_time + upload_time
        print(f"Task {task.id} on Node {node.id} → Exec: {exec_time:.2f}s, Upload: {upload_time:.2f}s, Total: {total:.2f}s")
    print("-" * 60)