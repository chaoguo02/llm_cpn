from core_function.update_time import computing_Task, computing_upload_time

def decode1_greedy(individual, task, nodes, taskflows, pset):
    try:
        # 为 task 选择执行总耗时（计算 + 上传）最小的节点
        return min(nodes, key=lambda n: computing_Task(task, n) + computing_upload_time(task, n))
    except Exception as e:
        print(f"[❌ 异常] decode1_greedy 处理任务 {task.global_id} 时出错：{e}")
        return None


def decode2_greedy(individual, node, taskflows, nodes, pset):
    try:
        # 从该节点的等待队列中选择执行总耗时（计算 + 上传）最小的任务
        return min(node.waiting_queue, key=lambda t: computing_Task(t, node) + computing_upload_time(t, node))
    except Exception as e:
        print(f"[❌ 异常] decode2_greedy 选择节点 {node.id} 的等待任务时出错：{e}")
        return None
