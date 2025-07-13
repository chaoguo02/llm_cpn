from core_function.update_time import computing_Task

# decode1_sjf：为当前任务选择预计计算时间最短的节点
def decode1_sjf(individual, task, nodes, taskflows, pset):
    try:
        return min(nodes, key=lambda n: computing_Task(task, n))
    except Exception as e:
        print(f"[❌ 异常] decode1_sjf 处理任务 {task.global_id} 时出错：{e}")
        return None

# decode2_sjf：从等待队列中选择计算时间最短的任务
def decode2_sjf(individual, node, taskflows, nodes, pset):
    try:
        return min(node.waiting_queue, key=lambda t: computing_Task(t, node))
    except Exception as e:
        print(f"[❌ 异常] decode2_sjf 选择节点 {node.id} 的等待任务时出错：{e}")
        return None