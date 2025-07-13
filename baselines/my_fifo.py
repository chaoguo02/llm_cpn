from core_function.update_time import predict_node_idle_time

def get_fifo_node_score(node):
    return -node.begin_idle_time

def get_fifo_task_score(task):
    return -task.arrivetime

# def decode1_fifo(individual, task, nodes, taskflows, pset):
#     try:
#         # FIFO 策略：选择最早空闲的节点
#         available_nodes = sorted(nodes, key=lambda n: n.begin_idle_time)
#         if not available_nodes:
#             return None
#         return available_nodes[0]
#     except Exception as e:
#         print(f"[❌ 异常] decode1_fifo 处理任务 {task.global_id} 时出错：{e}")
#         return None
#
# def decode1_fifo(individual, task, nodes, taskflows, pset):
#     try:
#         # 任务到达时不能“穿越时间”执行，只能等节点空闲
#         # 所以我们找出能最早为当前任务服务的节点（基于max(任务到达时间, 节点空闲时间)）
#         node_scores = []
#         for node in nodes:
#             start_time = max(task.present_time, node.begin_idle_time)
#             node_scores.append((node, start_time))
#
#         # 选择 earliest start_time 的节点
#         best_node = min(node_scores, key=lambda x: x[1])[0]
#         return best_node
#     except Exception as e:
#         print(f"[❌ 异常] decode1_fifo 处理任务 {task.global_id} 时出错：{e}")
#         return None

def decode1_fifo(individual, task, nodes, taskflows, pset):
    try:
        # 选择预测总空闲时间最早的节点
        return min(nodes, key=predict_node_idle_time)
    except Exception as e:
        print(f"[❌ 异常] decode1_fifo 处理任务 {task.global_id} 时出错：{e}")
        return None


def decode2_fifo(individual, node, taskflows, nodes, pset):
    try:
        # 选择等待队列中，arrivetime最早的任务
        return min(node.waiting_queue, key=lambda t: t.arrivetime)
    except Exception as e:
        print(f"[❌ 异常] decode2_fifo 选择节点 {node.id} 的等待任务时出错：{e}")
        return None
