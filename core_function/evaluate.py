import copy
import multiprocessing

from core_function.update_time import (
    present_time_update, find_earlist_time,
    computing_Task, computing_upload_time, update_next_free_time
)

def evaluate_offspring_in_parallel(offspring, taskflows, nodes, pset,decode1,decode2, work_processing,return_log=False):
    pool = multiprocessing.Pool()
    results = []

    for ind in offspring:
        tasks_copy = copy.deepcopy(taskflows)
        nodes_copy = copy.deepcopy(nodes)
        result = pool.apply_async(work_processing, (ind, tasks_copy, nodes_copy, pset,decode1,decode2,return_log))
        results.append(result)

    pool.close()
    pool.join()

    fitnesses = [res.get() for res in results]
    return fitnesses


def evaluate_on_testSets(individual, nodes, pset,decode1,decode2, pre_generated_taskflows, return_log=False):
    results = []
    logs = []

    for i, taskflows in enumerate(pre_generated_taskflows):
        nodes_copy = copy.deepcopy(nodes)
        taskflows_copy = copy.deepcopy(taskflows)

        if return_log:
            fitness, log_data = work_processing(individual, taskflows_copy, nodes_copy, pset,decode1,decode2, return_log=True)
            logs.append({
                "test_set_index": i,
                "fitness": fitness,
                "log": log_data
            })
        else:
            fitness = work_processing(individual, taskflows_copy, nodes_copy, pset,decode1, decode2, return_log)
            results.append(fitness[0])

    return (results, logs) if return_log else results

def sanitize_task_id(task):
    return getattr(task, "global_id", f"Task {task.id}")

def sanitize_node_id(node):
    return f"N{node.id}({node.node_type})"

def log_task_execution(log, task, node, start_time, end_time):
    log.append({
        "task_id": sanitize_task_id(task),
        "taskflow_id": task.taskflow_id,
        "node_id": sanitize_node_id(node),
        "start_time": start_time,
        "end_time": end_time
    })

def log_taskflow_summary(summary_log, tf):
    duration = tf.finish_time - tf.all_arrive_time
    summary_log.append({
        "taskflow_id": tf.id,
        "start_time": tf.all_arrive_time,
        "end_time": tf.finish_time,
        "duration": duration
    })

def summarize_taskflows(taskflows, taskflow_summary_log):
    sum_time, count = 0, 0
    for tf in taskflows:
        log_taskflow_summary(taskflow_summary_log, tf)
        sum_time += tf.finish_time - tf.all_arrive_time
        count += 1
    avg_time = sum_time / count if count else 0
    return avg_time


def work_processing(individual, taskflows, nodes, pset,decode1, decode2, return_log=False):
    task_execution_log, taskflow_summary_log, skipped_tasks_log = [], [], []
    node_assignment_log, queue1, queue2 = {}, [], []

    present_time = 0
    present_time_update(present_time, taskflows)
    print("\n🚀 [调度开始] 模拟任务调度流程启动...")

    for taskflow in taskflows:
        tasks = taskflow.find_predecessor_is_zero()
        for task in tasks:
            queue1.append((task, task.arrivetime))

    while queue1 or queue2:
        queue1, queue2, task_queue1, task_queue2 = find_earlist_time(queue1, queue2)

        if task_queue1:
            current_time = queue1[0][1]
            present_time_update(present_time=current_time, taskflows=taskflows)

            for task in task_queue1:
                node = decode1(individual, task, nodes, taskflows, pset)
                task.node = node

                if task.present_time >= node.begin_idle_time:
                    allocate_resources(node, task)
                    task_time = computing_Task(task, node) + computing_upload_time(task, node)
                    endtime = task.present_time + task_time
                    task.endtime = endtime
                    queue2.append((task, endtime))
                    node.begin_idle_time = endtime
                    node.next_free_time = update_next_free_time(node,task.present_time)  # 遍历等待队列得到的预期空闲时间
                    log_task_execution(task_execution_log,task,node,task.present_time,task.endtime)

                else:
                    node.waiting_queue.append(task)
                    node.next_free_time = update_next_free_time(node,task.present_time)  # 遍历等待队列得到的预期空闲时间

                if sanitize_node_id(node) not in node_assignment_log:
                    node_assignment_log[sanitize_node_id(node)] = []
                node_assignment_log[sanitize_node_id(node)].append(sanitize_task_id(task))

                queue1 = [item for item in queue1 if item[0] != task]

        if task_queue2:
            current_time = queue2[0][1]
            present_time_update(present_time=current_time, taskflows=taskflows)

            for finish_event in task_queue2:
                finish_event.finish = True
                finish_event.node.last_task_endtime = finish_event.endtime
                release_resources(finish_event.node, finish_event)

            for task in task_queue2:
                taskflow = taskflows[task.taskflow_id]
                if task.descendant:
                    current_index = taskflow.tasks.index(task)
                    descendant_tasks = []
                    for d in task.descendant:
                        taskflow.tasks[d].predecessor.remove(current_index)
                        if not taskflow.tasks[d].predecessor:
                            descendant_tasks.append(taskflow.tasks[d])
                    for descendant_task in descendant_tasks:
                        queue1.append((descendant_task, descendant_task.present_time))
                else:
                    taskflow.finish_time = max(taskflow.finish_time, current_time)

                if task.node.waiting_queue:
                    next_task = decode2(individual, task.node, taskflows, nodes, pset)
                    if next_task is None:
                        continue
                    task.node.waiting_queue.remove(next_task)
                    allocate_resources(task.node, next_task)
                    # 为什么是task，不是next_task
                    trans_delay = 0.1 * computing_upload_time(task, task.node)
                    task_time = computing_Task(next_task, task.node) + computing_upload_time(next_task, task.node) + trans_delay
                    next_task.endtime = next_task.present_time + task_time
                    queue2.append((next_task, next_task.endtime))
                    task.node.begin_idle_time = next_task.endtime
                    update_next_free_time(task.node, next_task.present_time)
                    log_task_execution(task_execution_log,next_task,task.node,next_task.present_time,next_task.endtime)

                else:
                    task.node.next_free_time = task.node.begin_idle_time

                queue2 = [item for item in queue2 if item[0] != task]

    avg_time = summarize_taskflows(taskflows, task_execution_log)
    print(f"\n📈 任务流平均完成时间：{avg_time:.2f}")

    if return_log:
        log_data = {
            "avg_time": avg_time,
            "task_execution_log": task_execution_log,
            "node_assignment_log": node_assignment_log,
            "taskflow_summary_log": taskflow_summary_log,
            "skipped_tasks_log": skipped_tasks_log
        }
        return avg_time, log_data
    return (avg_time,)


def allocate_resources(node, task):
#     node.cpu_capacity -= task.cpu_require
#     node.ram_capacity -= task.ram_require
#     node.gpu_capacity -= task.gpu_require
    pass
#
def release_resources(node, task):
#     node.cpu_capacity += task.cpu_require
#     node.ram_capacity += task.ram_require
#     node.gpu_capacity += task.gpu_require
    pass