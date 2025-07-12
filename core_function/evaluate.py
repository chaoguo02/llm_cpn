import copy
import multiprocessing

from core_function.update_time import present_time_update, find_earlist_time, computing_Task, computing_upload_time

def evaluate_offspring_in_parallel(offspring, taskflows, nodes, pset, work_processing):
    pool = multiprocessing.Pool()
    results = []

    for ind in offspring:
        tasks_copy = copy.deepcopy(taskflows)
        nodes_copy = copy.deepcopy(nodes)
        result = pool.apply_async(work_processing, (ind, tasks_copy, nodes_copy, pset,))
        results.append(result)

    pool.close()
    pool.join()

    fitnesses = [res.get() for res in results]
    return fitnesses

def evaluate_on_testSets(individual, nodes, pset, pre_generated_taskflows, return_log=False):
    results = []
    logs = []

    for i, taskflows in enumerate(pre_generated_taskflows):
        nodes_copy = copy.deepcopy(nodes)
        taskflows_copy = copy.deepcopy(taskflows)

        if return_log:
            fitness, log_data = work_processing(individual, taskflows_copy, nodes_copy, pset, return_log=True)
            logs.append({
                "test_set_index": i,
                "fitness": fitness,
                "log": log_data
            })
        else:
            fitness = work_processing(individual, taskflows_copy, nodes_copy, pset)
        results.append(fitness[0])  # fitness is a tuple: (avg_time,)

    if return_log:
        return results, logs
    else:
        return results


def work_processing(individual, taskflows, nodes, pset, return_log=False):
    def sanitize_task_id(task):
        return getattr(task, "global_id", f"Task {task.id}")

    def sanitize_node_id(node):
        return f"N{node.id}({node.node_type})"

    task_execution_log = []
    node_assignment_log = {}
    taskflow_summary_log = []
    skipped_tasks_log = []

    queue1 = []  # 未执行任务队列
    queue2 = []  # 正在执行的任务队列
    present_time = 0
    present_time_update(present_time, taskflows)

    print("🚀 [调度开始] 模拟任务调度流程启动...\n")

    for taskflow in taskflows:
        tasks = taskflow.find_predecessor_is_zero()
        for task in tasks:
            queue1.append((task, task.arrivetime))

    while queue1 or queue2:
        queue1, queue2, task_queue1, task_queue2 = find_earlist_time(queue1, queue2)

        if task_queue1:
            current_time = queue1[0][1]
            present_time_update(present_time=current_time, taskflows=taskflows)
            print(f"\n⏰ 时间推进至 {current_time:.2f}，处理队列 queue1 中的任务：")

            for task in task_queue1:
                node = decode1(individual, task, nodes, taskflows, pset)
                task.node = node
                print(f"🟡 任务 {sanitize_task_id(task)} 分配至节点 {sanitize_node_id(node)}")

                if task.present_time >= node.begin_idle_time:
                    try:
                        allocate_resources(task, node)
                    except Exception as e:
                        print(f"[❌资源分配失败] Task {task.global_id} 分配到 Node {node.id} 时失败：{e}")
                        skipped_tasks_log.append({
                            "task_id": sanitize_task_id(task),
                            "taskflow_id": task.taskflow_id,
                            "reason": f"资源分配失败: {str(e)}",
                            "node_id": sanitize_node_id(node),
                            "present_time": task.present_time
                        })
                        continue

                    task_time = computing_Task(task, node) + computing_upload_time(task, node)
                    endtime = task.present_time + task_time
                    task.endtime = endtime
                    queue2.append((task, endtime))
                    node.begin_idle_time = endtime

                    # ✅ 记录成功调度的任务
                    task_execution_log.append({
                        "task_id": sanitize_task_id(task),
                        "taskflow_id": task.taskflow_id,
                        "node_id": sanitize_node_id(node),
                        "start_time": task.present_time,
                        "end_time": task.endtime
                    })

                    print(f"✅ 执行任务 {sanitize_task_id(task)}，预计完成时间为 {endtime:.2f}")
                else:
                    node.waiting_queue.append(task)
                    print(f"⏳ 节点忙，任务 {sanitize_task_id(task)} 加入等待队列")

                if sanitize_node_id(node) not in node_assignment_log:
                    node_assignment_log[sanitize_node_id(node)] = []
                node_assignment_log[sanitize_node_id(node)].append(sanitize_task_id(task))

                queue1 = [item for item in queue1 if item[0] != task]

        if task_queue2:
            current_time = queue2[0][1]
            present_time_update(present_time=current_time, taskflows=taskflows)
            print(f"\n🏁 时间推进至 {current_time:.2f}，处理完成的任务：")

            for finish_event in task_queue2:
                finish_event.finish = True
                release_resources(finish_event, finish_event.node)
                print(f"✔️ 任务 {sanitize_task_id(finish_event)} 执行完成")

            for task in task_queue2:
                taskflow = taskflows[task.taskflow_id]

                if task.descendant:
                    current_index = taskflow.tasks.index(task)
                    descendant_tasks = []
                    for d in task.descendant:
                        taskflow.tasks[d].predecessor.remove(current_index)
                        if len(taskflow.tasks[d].predecessor) == 0:
                            descendant_tasks.append(taskflow.tasks[d])
                    for descendant_task in descendant_tasks:
                        queue1.append((descendant_task, descendant_task.present_time))
                        print(f"➡️ 后继任务 {sanitize_task_id(descendant_task)} 所有前驱完成，加入 queue1")
                else:
                    taskflow.finish_time = max(taskflow.finish_time, current_time)
                    print(f"🏁 任务流 {task.taskflow_id} 更新完成时间为 {taskflow.finish_time:.2f}")

                if task.node.waiting_queue:
                    next_task = decode2(individual, task.node, taskflows, nodes, pset)
                    task.node.waiting_queue.remove(next_task)
                    trans_delay = 0.1 * computing_upload_time(task, task.node)
                    task_time = computing_Task(next_task, task.node) + computing_upload_time(next_task, task.node) + trans_delay
                    next_task.endtime = next_task.present_time + task_time
                    queue2.append((next_task, next_task.endtime))
                    task.node.begin_idle_time = next_task.endtime

                    # ✅ 记录 decode2 分配的任务
                    task_execution_log.append({
                        "task_id": sanitize_task_id(next_task),
                        "taskflow_id": next_task.taskflow_id,
                        "node_id": sanitize_node_id(task.node),
                        "start_time": next_task.present_time,
                        "end_time": next_task.endtime
                    })

                    print(f"📤 节点 {sanitize_node_id(task.node)} 执行等待任务 {sanitize_task_id(next_task)}，完成时间为 {next_task.endtime:.2f}")

                queue2 = [item for item in queue2 if item[0] != task]

    print("\n📦 [任务分配统计] 各节点执行的任务分布如下：")
    for node in nodes:
        executed_tasks = node_assignment_log.get(sanitize_node_id(node), [])
        if executed_tasks:
            print(f"📌 节点 {sanitize_node_id(node)} 执行任务：{', '.join(executed_tasks)}")
        else:
            print(f"📌 节点 {sanitize_node_id(node)} 未执行任何任务")

    print("\n📊 [调度完成] 开始统计任务流完成时间：")
    sum_time = 0
    count = 0
    for tf in taskflows[10:]:
        duration = tf.finish_time - tf.all_arrive_time
        taskflow_summary_log.append({
            "taskflow_id": tf.id,
            "start_time": tf.all_arrive_time,
            "end_time": tf.finish_time,
            "duration": duration
        })
        print(f"📘 TaskFlow {tf.id}：开始 {tf.all_arrive_time:.2f}，完成 {tf.finish_time:.2f}，耗时 {duration:.2f}")
        sum_time += duration
        count += 1

    avg_time = sum_time / count if count > 0 else 0
    print(f"\n📈 后 10 个任务流的平均完成时间为：{avg_time:.2f}\n")

    if return_log:
        log_data = {
            "avg_time": avg_time,
            "task_execution_log": task_execution_log,
            "node_assignment_log": node_assignment_log,
            "taskflow_summary_log": taskflow_summary_log,
            "skipped_tasks_log": skipped_tasks_log
        }
        return avg_time, log_data
    else:
        return (avg_time,)


def allocate_resources(node, task):
    node.cpu_capacity -= task.cpu_require
    node.ram_capacity -= task.ram_require
    node.gpu_capacity -= task.gpu_require

def release_resources(node, task):
    node.cpu_capacity += task.cpu_require
    node.ram_capacity += task.ram_require
    node.gpu_capacity += task.gpu_require