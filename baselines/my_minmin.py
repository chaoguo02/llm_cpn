from core_function.update_time import estimated_finish_time


def decode1_minmin(individual, task, nodes, taskflows, pset):
    try:
        return min(nodes, key=lambda n: estimated_finish_time(task, n))

    except Exception as e:
        print(f"[❌ 异常] decode1_minmin 处理任务 {task.global_id} 时出错：{e}")
        return None

def decode2_minmin(individual, node, taskflows, nodes, pset):
    try:
        return min(node.waiting_queue, key=lambda t: estimated_finish_time(t, node))

    except Exception as e:
        print(f"[❌ 异常] decode2_minmin 选择节点 {node.id} 的等待任务时出错：{e}")
        return None