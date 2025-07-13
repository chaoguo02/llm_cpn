from deap import gp

from baselines.my_fifo import get_fifo_node_score, get_fifo_task_score
from core_function.update_time import computing_Task, computing_upload_time


def decode1(individual, task, nodes, taskflows, pset):
    try:
        heuristic_1 = gp.compile(expr=individual[0], pset=pset[0])
        scores = []
        for node in nodes:

            score = heuristic_1(computing_Task(task, node), task.present_time - task.arrivetime,
                len(task.descendant), computing_upload_time(task, node), get_fifo_node_score(node))
            scores.append((node, score))
        if not scores:
            return None
        return max(scores, key=lambda x: x[1])[0]
    except Exception as e:
        print(f"[❌ 异常] decode1 处理任务 {task.global_id} 时出错：{e}")
        return None

def decode2(individual, node, taskflows, nodes, pset):
    try:
        heuristic_2 = gp.compile(expr=individual[1], pset=pset[1])
        scores = []
        for task in node.waiting_queue:

            k = task.taskflow_id
            score = heuristic_2(computing_Task(task, node), task.present_time - task.arrivetime, len(task.descendant),
                taskflows[k].find_descendant_avg_time(taskflows, task, nodes),
                0.1 * computing_upload_time(task, node), get_fifo_task_score(task))
            scores.append((task, score))
        if not scores:
            return None
        return max(scores, key=lambda x: x[1])[0]
    except Exception as e:
        print(f"[❌ 异常] decode2 选择节点 {node.id} 的等待任务时出错：{e}")
        return None