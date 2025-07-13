from core_function.update_time import estimated_finish_time


def decode1_minmax(individual, task, nodes, taskflows, pset):
    try:
        # 找出完成时间最大的节点中的最小值节点
        worst_finish_time = [(n, estimated_finish_time(task, n)) for n in nodes]
        selected_node = max(worst_finish_time, key=lambda x: x[1])[0]
        return selected_node
    except Exception as e:
        print(f"[❌ 异常] decode1_minmax 处理任务 {task.global_id} 时出错：{e}")
        return None

def decode2_minmax(individual, node, taskflows, nodes, pset):
    try:
        # 节点等待队列中选择预估完成时间最长的任务
        if not node.waiting_queue:
            return None
        return max(node.waiting_queue, key=lambda t: estimated_finish_time(t, node))
    except Exception as e:
        print(f"[❌ 异常] decode2_minmax 选择节点 {node.id} 的等待任务时出错：{e}")
        return None