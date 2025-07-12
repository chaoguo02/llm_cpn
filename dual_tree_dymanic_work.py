import operator
import random
import numpy as np
from deap import base, creator, tools, gp
import networkx as nx
from deap.gp import graph
from networkx.drawing.nx_agraph import graphviz_layout
from collections import OrderedDict
# import matplotlib.pyplot as plt
from liu.DatasetReader import createNode
from Class.Taskflow import TaskFlow
from liu.exetime import computing_Task, computing_upload_time
import multiprocessing
import copy
import json
import os

#  统一参数设置
NUM_NODES = 5                 # 节点数
NUM_TASKFLOWS = 10             # 任务流数
POP_SIZE = 10                  # 种群规模
NGEN = 2                       # 进化代数
CXPB = 0.8                     # 交叉概率
MUTPB = 0.1                    # 变异概率
TOURNAMENT_SIZE = 1           # 锦标赛选择规模
HEIHT_LIMIT = 6               # 最大树高
NUM_TREES = 2                 # 每个个体中树的数量
NUM_RUNS = 1                # 重复运行次数（用于统计均值）
ELITISM_NUM = int(POP_SIZE * 0.05)  # 精英个体的数量
NUM_TEST_SETS = 1             # 测试集数量
NUM_TRAIN_SETS = 1             # 训练集数量
BASE_SEED = 10000              # 训练集我们从10000开始

def initIndividual(container, func, pset, size):
    return container(gp.PrimitiveTree(func(pset[i])) for i in range(size))

def cxOnePointListOfTrees(ind1, ind2):
    print("type:", type(ind1))
    for idx, (tree1, tree2) in enumerate(zip(ind1, ind2)):
        print(f"\n===== 交叉处理第 {idx} 棵子树 =====")
        print("🔵 原始 tree1 (str):", str(tree1))
        print("🔵 原始 tree2 (str):", str(tree2))

        HEIGHT_LIMIT = 8
        dec = gp.staticLimit(key=operator.attrgetter("height"), max_value=HEIGHT_LIMIT)
        tree1, tree2 = dec(gp.cxOnePoint)(tree1, tree2)
        print("🟢 交叉后 tree1_new (str):", str(tree1))
        print("🟢 交叉后 tree2_new (str):", str(tree2))
        ind1[idx], ind2[idx] = tree1, tree2

    return ind1, ind2

def mutUniformListOfTrees(individual, expr, pset):
    for idx, tree in enumerate(individual):
        HEIGHT_LIMIT = 8
        dec = gp.staticLimit(key=operator.attrgetter("height"), max_value=HEIGHT_LIMIT)
        tree, = dec(gp.mutUniform)(tree, expr=expr,pset=pset[idx])
        individual[idx] = tree

    return (individual,)

def varAnd(population, toolbox, cxpb, mutpb):
    offspring = [toolbox.clone(ind) for ind in population]
    evlpb=cxpb/(cxpb+mutpb)
    if random.random() < evlpb:
        for i in range(1, len(offspring), 2):
            offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1],
                                                          offspring[i])

            del offspring[i - 1].fitness.values, offspring[i].fitness.values

    else:
        for i in range(len(offspring)):
            offspring[i], = toolbox.mutate(offspring[i])
            del offspring[i].fitness.values

    return offspring

def keyadd(p,x):
    all_keys = set(p.keys()).union(x.keys())
    result = OrderedDict()
    for key in all_keys:
        result[key] = p.get(key, 0) + x.get(key, 0)
    return result

def sortPopulation(toolbox, population):
    # 克隆个体，避免修改原始种群
    populationCopy = [toolbox.clone(ind) for ind in population]
    # 使用 sorted 按适应度升序排序
    sorted_population = sorted(populationCopy, key=lambda ind: ind.fitness.values)
    return sorted_population

def is_number(string):#创建一个函数 is_number，用于检查一个字符串是否可以转换为浮点数。如果可以，则返回 True，否则返回 False。
    try:
        float(string)
        return True
    except ValueError:
        return False

def count_leaf_types(tree):
    leaf_count={}
    name_mapping = {
        'ARG0': 'a', 'ARG1': 'b', 'ARG2': 'c',
        'ARG3': 'd', 'ARG4': 'e', 'ARG5': 'f'
    }
    """统计每种叶子节点的数量"""
    for x in tree:
        if isinstance(x, gp.Terminal):  # 如果是叶子节点
            leaf_name = x.name
            if leaf_name in name_mapping:
                leaf_name = name_mapping[leaf_name]
            leaf_count[leaf_name] = leaf_count.get(leaf_name, 0) + 1
    return leaf_count

def protect_div(left, right):
    with np.errstate(divide='ignore', invalid='ignore'):
        x = np.divide(left, right)
        if isinstance(x, np.ndarray):
            x[np.isinf(x)] = 1
            x[np.isnan(x)] = 1
        elif np.isinf(x) or np.isnan(x):
            x = 1
    return x

def record_best_individual_log(individual, pre_generated_taskflows, nodes, pset, generation_index, run_index, base_dir="best_ind_logs"):

    test_taskflows_sample = copy.deepcopy(pre_generated_taskflows[0])
    nodes_log = copy.deepcopy(nodes)

    _, log_dict = work_processing(individual, test_taskflows_sample, nodes_log, pset, return_log=True)

    log_dict["individual_expressions"] = [str(tree) for tree in individual]

    run_dir = os.path.join(base_dir, f"run_{run_index}")
    os.makedirs(run_dir, exist_ok=True)

    log_path = os.path.join(run_dir, f"gen_{generation_index}_log.json")
    with open(log_path, "w", encoding='utf-8') as f:
        json.dump(log_dict, f, indent=2, ensure_ascii=False)

def get_fifo_node_score(node):
    """返回 FIFO 节点优先级得分：节点空闲时间越早得分越高"""
    return -node.begin_idle_time

def get_fifo_task_score(task):
    """返回 FIFO 任务优先级得分：到达时间越早得分越高"""
    return -task.arrivetime

def allocate_resources(node, task):
    node.cpu_capacity -= task.cpu_require
    node.ram_capacity -= task.ram_require
    node.gpu_capacity -= task.gpu_require

def release_resources(node, task):
    node.cpu_capacity += task.cpu_require
    node.ram_capacity += task.ram_require
    node.gpu_capacity += task.gpu_require

def createTaskFlows(num_TaskFlow,genre,seed): #创建多个工作流 #k为任务流随机种子
    lambda_rate = 1  # 平均到达速率 (λ，每单位时间平均到达任务数)
    np.random.seed(seed)
    interarrival_times = np.random.exponential(1 / lambda_rate, num_TaskFlow)  # 会生成一个包含 num_tasks 个任务到达时间间隔的数组

    # 计算每个任务的到达时间（通过累加间隔时间）
    arrival_times = np.cumsum(interarrival_times)
    taskflows=[TaskFlow(id,arrival_time,genre,id+(seed)*num_TaskFlow) for id, arrival_time in zip(range(num_TaskFlow), arrival_times)]
    return taskflows

def present_time_update(present_time,taskflows):#只更新那些未完成任务的当前时间，所以当一个任务已完成时，它的当前时间就是完成时间
    for taskflow in taskflows:
        for task in taskflow.tasks:
            if task.finish is False:
                task.present_time=present_time

def find_earlist_time(queue1,queue2):
    # 使用 sort() 方法排序
    queue1.sort(key=lambda x: x[1])
    queue2.sort(key=lambda x: x[1])
    task_queue1 = []
    task_queue2 = []
    if len(queue1)!=0 and len(queue2)!=0:
        min_time=min(queue1[0][1],queue2[0][1])
        for event in queue1:
            if event[1]==min_time:
                task_queue1.append(event[0])
            elif event[1] > min_time:
                break
        for event in queue2:
            if event[1]==min_time:
                task_queue2.append(event[0])
            elif event[1] > min_time:
                break
    elif len(queue1)==0  and len(queue2)!=0:
        min_time = queue2[0][1]
        for event in queue2:
            if event[1]==min_time:
                task_queue2.append(event[0])
            elif event[1] > min_time:
                break
    elif len(queue1)!=0  and len(queue2)==0:
        min_time = queue1[0][1]
        for event in queue1:
            if event[1]==min_time:
                task_queue1.append(event[0])
            elif event[1] > min_time:
                break
    return queue1,queue2,task_queue1,task_queue2

# 添加新方法

def decode1(individual, task, nodes, taskflows, pset):
    try:
        heuristic_1 = gp.compile(expr=individual[0], pset=pset[0])
        scores = []
        for node in nodes:
            if (node.cpu_available >= task.cpu_require and node.ram_available >= task.ram_require and
                node.gpu_available >= task.gpu_require):
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
            if (node.cpu_available >= task.cpu_require and node.ram_available >= task.ram_require and
                node.gpu_available >= task.gpu_require):
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

def eaSimple(population, toolbox, nodes, pre_generated_taskflows, num_TaskFlow,
             cxpb, mutpb, ngen, elitism, pset, num_run, min_fitness_values=None, genre_min_fitness_values=None):
    if min_fitness_values is None:
        min_fitness_values = []
    if genre_min_fitness_values is None:
        genre_min_fitness_values = []

    # 初始种群评估（这里也是要对训练集进行改造的）
    initial_inds = [ind for ind in population if not ind.fitness.valid]
    initial_taskflows_list = [
        createTaskFlows(num_TaskFlow, 0, BASE_SEED + i)
        for i in range(NUM_TRAIN_SETS)
    ]
    fitnesses_list = [
        evaluate_offspring_in_parallel(initial_inds, taskflows, nodes, pset, work_processing)
        for taskflows in initial_taskflows_list
    ]

    fitnesses_avg = [
        (np.mean([fitnesses_list[run][i][0] for run in range(NUM_TRAIN_SETS)]),)
        for i in range(len(initial_inds))
    ]

    for ind, fit in zip(initial_inds, fitnesses_avg):
        ind.fitness.values = fit

    elite_inds = sortPopulation(toolbox, population)[:elitism]

    # 测试集评估最优个体
    test_fitnesses = evaluate_on_testSets(elite_inds[0], nodes, pset, pre_generated_taskflows)
    genre_min_fitness_values.append(sum(test_fitnesses) / NUM_TEST_SETS)

    # 保存第0代最优个体调度日志和表达式
    record_best_individual_log(elite_inds[0],pre_generated_taskflows,nodes,pset,0,num_run)

    for gen in range(1, ngen + 1):

        # 每一轮训练的时候改为两次仿真
        # train_taskflows = createTaskFlows(num_TaskFlow, 0, gen)
        train_taskflows_list = [
            createTaskFlows(num_TaskFlow, 0, BASE_SEED + gen * NUM_TRAIN_SETS + i)
            for i in range(NUM_TRAIN_SETS)
        ]

        # 选择 + 变异 + 交叉
        offspring_inds = toolbox.select(population, len(population) - elitism)
        offspring_inds = varAnd(offspring_inds, toolbox, cxpb, mutpb)
        offspring_inds[:] = elite_inds + offspring_inds

        # 训练集评估
        fitnesses_list = []
        for train_taskflows in train_taskflows_list:
            fitnesses = evaluate_offspring_in_parallel(offspring_inds, train_taskflows, nodes, pset, work_processing)
            fitnesses_list.append(fitnesses)
        # 两次仿真评估并取平均适应度
        fitnesses_avg = []
        for i in range(len(offspring_inds)):
            avg_fit = np.mean([fitnesses_list[run][i][0] for run in range(NUM_TRAIN_SETS)])
            fitnesses_avg.append((avg_fit,))

        # 更新个体适应度
        for ind, fit in zip(offspring_inds, fitnesses_avg):
            ind.fitness.values = fit

        population[:] = offspring_inds

        elite_inds = sortPopulation(toolbox, population)[:elitism]

        # 测试集评估最优个体
        test_fitnesses = evaluate_on_testSets(elite_inds[0], nodes, pset, pre_generated_taskflows)
        genre_min_fitness_values.append(sum(test_fitnesses) / NUM_TEST_SETS)

        # 保存当前代最优个体调度日志和表达式
        record_best_individual_log(elite_inds[0],pre_generated_taskflows,nodes,pset,gen,num_run)

    return population, genre_min_fitness_values, elite_inds[0]

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

def main_dual_tree(num_run):
    nodes = createNode(NUM_NODES,0)
    # taskflows=createTaskFlows(NUM_TASKFLOWS,0,0)
    pre_generated_taskflows = []
    for i in range(NUM_TEST_SETS):
        taskflow = createTaskFlows(NUM_TASKFLOWS,1,i)
        pre_generated_taskflows.append(taskflow)
    pset = []
    for idx in range(2):
        if idx == 0:
            pset1 = gp.PrimitiveSet("MAIN", 4)
            pset1.addPrimitive(operator.add, 2)
            pset1.addPrimitive(operator.sub, 2)
            pset1.addPrimitive(operator.mul, 2)
            pset1.addPrimitive(protect_div, 2)
            pset1.addPrimitive(operator.neg, 1)
            pset1.addPrimitive(max, 2)
            pset1.addPrimitive(min, 2)
            pset1.renameArguments(ARG0='TCT')  # 任务的计算时间 tct
            pset1.renameArguments(ARG1='TWT')  # 任务的等待时间=任务的到达时间-任务的当前时间 twt
            pset1.renameArguments(ARG2='TNC')  # 任务的后继数量 tnc
            pset1.renameArguments(ARG3='TUT')  # 任务的上传时间  tut
            pset1.renameArguments(ARG4='FIFO1')
            pset.append(pset1)

        elif idx == 1:
            pset2 = gp.PrimitiveSet("MAIN", 5)
            pset2.addPrimitive(operator.add, 2)
            pset2.addPrimitive(operator.sub, 2)
            pset2.addPrimitive(operator.mul, 2)
            pset2.addPrimitive(protect_div, 2)
            pset2.addPrimitive(operator.neg, 1)
            pset2.addPrimitive(max, 2)
            pset2.addPrimitive(min, 2)
            pset2.renameArguments(ARG0='TCT')  # 任务的计算时间
            pset2.renameArguments(ARG1='TWT')  # 任务的等待时间=任务的到达时间-任务的当前时间
            pset2.renameArguments(ARG2='TNC')  # 任务的后继数量
            pset2.renameArguments(ARG3='TPS')  # 任务的给后继传递消息的时间 tps
            pset2.renameArguments(ARG4='ACT')  # 后继节点的平均完成时间 act
            pset2.renameArguments(ARG5='FIFO2')
            pset.append(pset2)

    # 创建一个适应度类和个体类，个体由多棵树组成
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, min_=2, max_=4)
    toolbox.register("individual", initIndividual, creator.Individual, toolbox.expr, pset=pset, size=NUM_TREES)  # 假设我们创建2个树
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", work_processing)#不能绑定taskflows=taskflows,nodes=nodes
    toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_SIZE)
    toolbox.register("mate", cxOnePointListOfTrees)
    toolbox.register("mutate", mutUniformListOfTrees, expr=toolbox.expr, pset=pset)
    toolbox.register("compile", gp.compile)

    population = toolbox.population(n=POP_SIZE)
    pop,genre_min_fitness_values,elite= eaSimple(population=population,
                                                       toolbox=toolbox,
                                                       # taskflows=taskflows,
                                                       nodes=nodes,
                                                       pre_generated_taskflows=pre_generated_taskflows,
                                                       num_TaskFlow=NUM_TASKFLOWS,
                                                       cxpb=CXPB,
                                                       mutpb=MUTPB,
                                                       ngen=NGEN,
                                                       elitism=ELITISM_NUM,
                                                       pset=pset,
                                                       num_run = num_run,
                                                       min_fitness_values=[],
                                                       genre_min_fitness_values=[]
                                                 )
    leaf_ratio_result=[]
    for tree in elite:
        # 统计每种终端的数量
        leaf_count = count_leaf_types(tree)
        # 计算树中所有终端的总数
        total_leaves = sum(leaf_count.values())
        # 计算每种终端的比例，并将其存储在OrderedDict中
        leaf_ratio = OrderedDict((key, value / total_leaves) for key, value in leaf_count.items())
        leaf_ratio_result.append(leaf_ratio)
    return genre_min_fitness_values,leaf_ratio_result

if __name__ == '__main__':

    run_fitness_history = []
    # 记录每一代的测试集最小适应度的累加值
    genre_min_fitness_values_sum = [0] * NGEN
    # 记录每棵树的叶子类型统计比例累加值
    leaf_ratio_result_sum = [OrderedDict() for _ in range(NUM_TREES)]
    # NUM_RUNS 次独立运行
    for num_run in range(NUM_RUNS):
        # 每次独立运行后，得到每一代的测试集的最小适应度，以及每棵树的叶子比例统计
        genre_min_fitness_values, leaf_ratio_result = main_dual_tree(num_run)

        run_fitness_history.append(genre_min_fitness_values)
        genre_min_fitness_values_sum = [a + b for a, b in
                                        zip(genre_min_fitness_values, genre_min_fitness_values_sum)]
        leaf_ratio_result_sum = [keyadd(a, b) for a, b in zip(leaf_ratio_result, leaf_ratio_result_sum)]

    genre_min_fitness_values_sum = [a / NUM_RUNS for a in genre_min_fitness_values_sum]



