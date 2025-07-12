import operator
import random
import numpy as np
from deap import base, creator, tools, gp
import networkx as nx
from deap.gp import graph
from networkx.drawing.nx_agraph import graphviz_layout
from collections import OrderedDict

from core_function.data_loader import createNode
from core_function.update_time import computing_Task, computing_upload_time
# import matplotlib.pyplot as plt
from entity.Taskflow import TaskFlow
import multiprocessing
import copy
import json
import os

#  统一参数设置
NUM_NODES = 10                 # 节点数
NUM_TASKFLOWS = 50             # 任务流数
POP_SIZE = 100                  # 种群规模
NGEN = 50                       # 进化代数
CXPB = 0.8                     # 交叉概率
MUTPB = 0.1                    # 变异概率
TOURNAMENT_SIZE = 5           # 锦标赛选择规模
HEIHT_LIMIT = 8               # 最大树高
NUM_TREES = 2                 # 每个个体中树的数量
NUM_RUNS = 1                # 重复运行次数（用于统计均值）
ELITISM_NUM = int(POP_SIZE * 0.05)  # 精英个体的数量
NUM_TEST_SETS = 30             # 测试集数量
NUM_TRAIN_SETS = 2             # 训练集数量
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

def sortPopulation(toolbox, population):#函数 sortPopulation 对输入的种群进行排序（按照适应度,冒泡排序），并返回一个排序后的种群副本，而不会修改原始的population
    populationCopy = [toolbox.clone(ind) for ind in population]
    popsize = len(population)
    for j in range(popsize):
        sign = False
        for i in range(popsize-1-j):
            sum_fit_i = populationCopy[i].fitness.values
            sum_fit_i_1 = populationCopy[i+1].fitness.values
            if sum_fit_i > sum_fit_i_1:
                populationCopy[i], populationCopy[i+1] = populationCopy[i+1], populationCopy[i]
                sign = True
        if not sign:
            break
    return populationCopy

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

# def evaluate_on_testSets(individual, nodes, pset, pre_generated_taskflows):
#     results = []
#     for i,taskflows in enumerate(pre_generated_taskflows):
#         nodes_copy = copy.deepcopy(nodes)
#         taskflows = copy.deepcopy(taskflows)
#         fitness = work_processing(individual, taskflows, nodes_copy, pset)
#         results.append(fitness[0])
#     return results

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



def eaSimple(population, toolbox, taskflows, nodes, pre_generated_taskflows, num_TaskFlow, num_nodes,
             cxpb, mutpb, ngen, elitism, pset,
             min_fitness_values=None, genre_min_fitness_values=None,
             stats=None, halloffame=None, verbose=__debug__):
    if min_fitness_values is None:
        min_fitness_values = []
    if genre_min_fitness_values is None:
        genre_min_fitness_values = []

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # 初始种群评估（这里也是要对训练集进行改造的）
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = evaluate_offspring_in_parallel(invalid_ind, taskflows, nodes, pset, work_processing)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    min_fitness_values.append(record["fitness"]["min"])
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    sorted_elite = sortPopulation(toolbox, population)[:elitism]

    # 测试集评估最优个体
    value_list = evaluate_on_testSets(sorted_elite[0], nodes, pset, pre_generated_taskflows)
    genre_min_fitness_values.append(sum(value_list) / NUM_TEST_SETS)

    # 保存第0代最优个体调度日志和表达式
    taskflows_log = createTaskFlows(num_TaskFlow, 0, 0)
    nodes_log = copy.deepcopy(nodes)
    _, log_dict = work_processing(sorted_elite[0], taskflows_log, nodes_log, pset, return_log=True)
    os.makedirs("best_ind_logs", exist_ok=True)
    with open("best_ind_logs/gen_0_log.json", "w", encoding='utf-8') as f:
        json.dump(log_dict, f, indent=2, ensure_ascii=False)
    with open("best_ind_logs/gen_0_expr.txt", "w", encoding='utf-8') as f:
        f.write(str(sorted_elite[0]))

    for gen in range(1, ngen + 1):
        taskflows = createTaskFlows(num_TaskFlow, 0, gen)

        # 选择 + 变异 + 交叉
        offspring = toolbox.select(population, len(population) - elitism)
        offspring = varAnd(offspring, toolbox, cxpb, mutpb)
        offspring[:] = sorted_elite + offspring

        # 训练集评估
        fitnesses = evaluate_offspring_in_parallel(offspring, taskflows, nodes, pset, work_processing)
        for ind, fit in zip(offspring, fitnesses):
            ind.fitness.values = fit

        if halloffame is not None:
            halloffame.update(offspring)

        population[:] = offspring
        record = stats.compile(population) if stats else {}
        min_fitness_values.append(record["fitness"]["min"])
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)

        if verbose:
            print(logbook.stream)

        sorted_elite = sortPopulation(toolbox, population)[:elitism]

        # 测试集评估最优个体
        value_list = evaluate_on_testSets(sorted_elite[0], nodes, pset, pre_generated_taskflows)
        genre_min_fitness_values.append(sum(value_list) / NUM_TEST_SETS)

        # 保存当前代最优个体调度日志和表达式
        taskflows_log = createTaskFlows(num_TaskFlow, 0, gen)
        nodes_log = copy.deepcopy(nodes)
        _, log_dict = work_processing(sorted_elite[0], taskflows_log, nodes_log, pset, return_log=True)
        with open(f"best_ind_logs/gen_{gen}_log.json", "w", encoding='utf-8') as f:
            json.dump(log_dict, f, indent=2, ensure_ascii=False)
        with open(f"best_ind_logs/gen_{gen}_expr.txt", "w", encoding='utf-8') as f:
            f.write(str(sorted_elite[0]))

    return population, logbook, genre_min_fitness_values, sorted_elite[0]


# def eaSimple(population, toolbox,taskflows,nodes,pre_generated_taskflows,num_TaskFlow,num_nodes, cxpb, mutpb, ngen, elitism,pset,min_fitness_values=[],genre_min_fitness_values=[],stats=None,
#              halloffame=None, verbose=__debug__, ):
#     logbook = tools.Logbook()
#     logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
#
#
#     invalid_ind = [ind for ind in population if not ind.fitness.valid]
#
#     fitnesses = evaluate_offspring_in_parallel(invalid_ind, taskflows, nodes, pset, work_processing)
#
#     for ind, fit in zip(invalid_ind, fitnesses):
#         ind.fitness.values = fit
#     if halloffame is not None:
#         halloffame.update(population)
#
#     record = stats.compile(population) if stats else {}
#     min_fitness_values.append(record["fitness"]["min"])
#     logbook.record(gen=0, nevals=len(invalid_ind), **record)
#     if verbose:
#         print(logbook.stream)
#
#     sorted_elite = sortPopulation(toolbox, population)[:elitism]
#     # 适应度评估（验证集）
#     value_list = evaluate_on_testSets(sorted_elite[0], nodes, pset, pre_generated_taskflows)
#     genre_min_fitness_values.append(sum(value_list) / NUM_TEST_SETS)
#
#     # Begin the generational process
#     for gen in range(1, ngen + 1):
#         taskflows = createTaskFlows(num_TaskFlow,0,gen) #每一代都是全新的taskflow
#
#         # Select the next generation individuals
#         offspring = toolbox.select(population, len(population)-elitism) # 这行代码使用选择操作从当前种群中选择与原种群相同数量的个体，这些个体构成新的后代。,只是选择的过程为3选一
#
#         # Vary the pool of individuals
#         offspring = varAnd(offspring, toolbox, cxpb, mutpb)
#
#         offspring[:] = sorted_elite + offspring
#
#         # 适应度评估（训练集）
#         fitnesses = evaluate_offspring_in_parallel(offspring, taskflows, nodes, pset, work_processing)
#
#         for ind, fit in zip(offspring, fitnesses):
#             ind.fitness.values = fit   #将这些计算出的适应度值赋给对应的个体，使得它们的适应度属性变为有效。
#
#         # Update the hall of fame with the generated individuals
#         if halloffame is not None:
#             halloffame.update(offspring)
#
#         # Replace the current population by the offspring
#         population[:] = offspring
#
#         # Append the current generation statistics to the logbook
#         record = stats.compile(population) if stats else {}
#         min_fitness_values.append(record["fitness"]["min"]) #我加的
#         logbook.record(gen=gen, nevals=len(invalid_ind), **record)
#
#         if verbose:
#             print(logbook.stream)
#
#         sorted_elite = sortPopulation(toolbox, population)[:elitism]
#
#
#         # 适应度评估（测试集）
#         value_list = evaluate_on_testSets(sorted_elite[0], nodes, pset, pre_generated_taskflows)
#         genre_min_fitness_values.append(sum(value_list) / NUM_TEST_SETS)
#     return population, logbook ,genre_min_fitness_values,sorted_elite[0]

def is_number(string):#创建一个函数 is_number，用于检查一个字符串是否可以转换为浮点数。如果可以，则返回 True，否则返回 False。
    try:
        float(string)
        return True
    except ValueError:
        return False


# def plot_a_tree(tree):
#     red_nodes = []
#     purple_nodes = []
#     blue_nodes = []  # 创建三个空列表，用于存储不同颜色节点的索引。
#
#     for gid, g in enumerate(tree):
#         # 假设这里的 blue_nodes 是想要把所有节点都标记为蓝色
#         blue_nodes.append(gid)
#
#     # 假设 graph 函数已经定义好并返回 nodes, edges, labels
#     nodes, edges, labels = graph(tree)
#
#     # 修改 labels 中的操作符
#     for x in labels:
#         if labels[x] == "add":
#             labels[x] = "+"
#         elif labels[x] == "sub":
#             labels[x] = "-"
#         elif labels[x] == "mul":
#             labels[x] = "*"
#         elif labels[x] == "protected_div":
#             labels[x] = "/"
#
#     # 创建 NetworkX 图对象
#     g = nx.Graph()
#     g.add_nodes_from(nodes)
#     g.add_edges_from(edges)
#
#     # 计算每种节点颜色的索引
#     red_nodes_idx = [nodes.index(n) for n in nodes if n in red_nodes]
#     purple_nodes_idx = [nodes.index(n) for n in nodes if n in purple_nodes]
#     blue_nodes_idx = [nodes.index(n) for n in nodes if n in blue_nodes]
#
#     # 创建新图形对象
#     plt.figure(figsize=(10, 6))  # 每次绘制时创建新的图形对象
#
#     # 使用 graphviz_layout 布局，使节点按树形结构排列
#     pos = graphviz_layout(g, prog="dot")
#
#     # 绘制不同颜色的节点
#     nx.draw_networkx_nodes(g, pos, nodelist=red_nodes_idx, node_color="darkred", node_size=500, edgecolors='black', node_shape='o')
#     nx.draw_networkx_nodes(g, pos, nodelist=purple_nodes_idx, node_color="indigo", node_size=500, edgecolors='black', node_shape='o')
#     nx.draw_networkx_nodes(g, pos, nodelist=blue_nodes_idx, node_color="white", node_size=500, edgecolors='black', node_shape='o')
#
#     # 绘制边
#     nx.draw_networkx_edges(g, pos, edge_color='black')
#
#     # 绘制节点标签，字体颜色为黑色
#     nx.draw_networkx_labels(g, pos, labels, font_color="black", font_size=9)  # 将字体颜色设置为黑色，字体大小为9
#
#     # 确保目标目录存在
#     output_dir = r"D:\result\tree"
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#
#     # 使用当前时间戳生成唯一的文件名，避免覆盖
#     timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#     save_path = os.path.join(output_dir, f"tree_image_{timestamp}.pdf")
#
#     # 保存图像为PDF格式
#     plt.savefig(save_path, format='pdf', bbox_inches='tight')
#     plt.close()  # 确保每次都关闭当前图形


def count_leaf_types(tree):
    leaf_count={}
    name_mapping = {
        'ARG0': 'a',
        'ARG1': 'b',
        'ARG2': 'c',
        'ARG3': 'd',
        'ARG4': 'e',
        'ARG5': 'f'
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


def decode1(individual, task, nodes, taskflows,pset):#decode1函数中需要增加一个约束条件，判断节点的ram，gpu，cpu够不够用，要在够用的节点中选择
    heuristic_1 = gp.compile(expr=individual[0], pset=pset[0])
    scores = []
    for node in nodes:
        heuristic_score = heuristic_1(computing_Task(task,node),task.present_time-task.arrivetime,len(task.descendant),computing_upload_time(task,node))
        scores.append((node, heuristic_score))
    best_node = max(scores, key=lambda x: x[1])[0]
    return best_node #找到节点本身

def decode2(individual, node, taskflows,nodes, pset):
    heuristic_2 = gp.compile(expr=individual[1], pset=pset[1])
    scores = []
    for task in node.waiting_queue:
        k = task.taskflow_id#任务所在任务流，在多任务流中的位置
        heuristic_score = heuristic_2(computing_Task(task,node),task.present_time-task.arrivetime,len(task.descendant),
                                      0.1*computing_upload_time(task,node),taskflows[k].find_descendant_avg_time(taskflows,task,nodes)) #任务给后继传递消息的时间取上传时间的0.1
        scores.append((task, heuristic_score))
    best_task = max(scores, key=lambda x: x[1])[0]
    return best_task #返回找到的任务本身

def work_processing(individual, taskflows, nodes, pset, return_log=False):
    def sanitize_task_id(task):
        return getattr(task, "global_id", f"Task {task.id}")

    def sanitize_node_id(node):
        return f"N{node.id}({node.node_type})"

    task_execution_log = []
    node_assignment_log = {}
    taskflow_summary_log = []

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
                    task_time = computing_Task(task, node) + computing_upload_time(task, node)
                    endtime = task.present_time + task_time
                    task.endtime = endtime
                    queue2.append((task, endtime))
                    node.begin_idle_time = endtime
                    print(f"✅ 执行任务 {sanitize_task_id(task)}，预计完成时间为 {endtime:.2f}")
                else:
                    node.waiting_queue.append(task)
                    print(f"⏳ 节点忙，任务 {sanitize_task_id(task)} 加入等待队列")

                task_execution_log.append({
                    "task_id": sanitize_task_id(task),
                    "taskflow_id": task.taskflow_id,
                    "node_id": sanitize_node_id(node),
                    "start_time": task.present_time,
                    "end_time": task.endtime if hasattr(task, 'endtime') else None
                })

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
            "taskflow_summary_log": taskflow_summary_log
        }
        return avg_time, log_data
    else:
        return (avg_time,)



def random_value():
    return random.randint(-1, 1)
# 定义命名函数来替换 lambda
def get_fitness_values(individual):
    return individual.fitness.values

def get_max_tree_height(individual):
    return max([tree.height for tree in individual])

def main_dual_tree():
    nodes = createNode(NUM_NODES,0)
    taskflows=createTaskFlows(NUM_TASKFLOWS,0,0)
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
            #pset1.addPrimitive(np.divide, 2)
            pset1.addPrimitive(operator.neg, 1)
            pset1.addPrimitive(max, 2)
            pset1.addPrimitive(min, 2)
            #pset1.addEphemeralConstant("rand101", lambda: random.randint(-1, 1))

            pset1.renameArguments(ARG0='TCT')  # 任务的计算时间 tct
            pset1.renameArguments(ARG1='TWT')  # 任务的等待时间=任务的到达时间-任务的当前时间 twt
            pset1.renameArguments(ARG2='TNC')  # 任务的后继数量 tnc
            #pset1.renameArguments(ARG3='ACT')  # 后继节点的平均完成时间 act
            pset1.renameArguments(ARG3='TUT')  # 任务的上传时间  tut


            pset.append(pset1)

        elif idx == 1:
            pset2 = gp.PrimitiveSet("MAIN", 5)
            pset2.addPrimitive(operator.add, 2)
            pset2.addPrimitive(operator.sub, 2)
            pset2.addPrimitive(operator.mul, 2)
            pset2.addPrimitive(protect_div, 2)
            #pset2.addPrimitive(np.divide, 2)
            pset2.addPrimitive(operator.neg, 1)
            pset2.addPrimitive(max, 2)
            pset2.addPrimitive(min, 2)

            pset2.renameArguments(ARG0='TCT')  # 任务的计算时间
            pset2.renameArguments(ARG1='TWT')  # 任务的等待时间=任务的到达时间-任务的当前时间
            pset2.renameArguments(ARG2='TNC')  # 任务的后继数量
            #pset2.renameArguments(ARG3='NTQ')  # 第二个树队列中任务个数 ntq
            pset2.renameArguments(ARG3='TPS')  # 任务的给后继传递消息的时间 tps
            pset2.renameArguments(ARG4='ACT')  # 后继节点的平均完成时间 act
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
    hof = tools.HallOfFame(1)
    stats_fit = tools.Statistics(key=get_fitness_values)
    stats_size = tools.Statistics(key=get_max_tree_height)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)
    pop, log ,genre_min_fitness_values,elite= eaSimple(population=population,
                                                       toolbox=toolbox,
                                                       taskflows=taskflows,
                                                       nodes=nodes,
                                                       pre_generated_taskflows=pre_generated_taskflows,
                                                       num_TaskFlow=NUM_TASKFLOWS,
                                                       num_nodes=NUM_NODES,
                                                       cxpb=CXPB,
                                                       mutpb=MUTPB,
                                                       ngen=NGEN,
                                                       elitism=ELITISM_NUM,
                                                       pset=pset,
                                                       min_fitness_values=[],
                                                       genre_min_fitness_values=[],
                                                       stats=mstats,
                                                       halloffame=hof,
                                                       verbose=True)

    # 查看最佳个体
    best_ind = hof[0]
    print('Best individual fitness:', best_ind.fitness)
    leaf_ratio_result=[]
    for tree in elite:
        # plot_a_tree(tree)

        # 统计每种终端的数量
        leaf_count = count_leaf_types(tree)

        # 计算树中所有终端的总数
        total_leaves = sum(leaf_count.values())

        # 计算每种终端的比例，并将其存储在OrderedDict中
        leaf_ratio = OrderedDict((key, value / total_leaves) for key, value in leaf_count.items())
        leaf_ratio_result.append(leaf_ratio)
    return  genre_min_fitness_values,leaf_ratio_result


# def min_fitness_trend(min_fitness_values_1):
#     # 创建保存路径的文件夹（如果不存在）
#     save_folder = 'D:/result/picture'
#     if not os.path.exists(save_folder):
#         os.makedirs(save_folder)
#
#     # 更新字体大小
#     plt.rcParams.update({'font.size': 20})
#
#     # 设置图形的大小
#     plt.figure(figsize=(10, 6))
#     plt.grid(True)
#
#     # 横坐标为次数，假设为列表的索引
#     x = list(range(len(min_fitness_values_1)))  # 横坐标
#
#     # 绘制各条曲线
#     plt.plot(x, min_fitness_values_1, label='DTGP', marker='o')
#
#     # 添加标签和标题
#     plt.xlabel('Generation')
#     plt.ylabel('Makespan(ms)')
#
#     # 调整图例的位置，防止遮挡
#     plt.legend(loc='upper right')
#
#     # 使用当前时间戳生成唯一的文件名，避免覆盖
#     timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#     save_path = os.path.join(save_folder, f"fitness_plot_{timestamp}.pdf")
#
#     # 保存图像，并裁剪掉周围的空白区域
#     plt.savefig(save_path, format='pdf', bbox_inches='tight')
#     plt.close()  # 关闭图形，防止图形在显示时被多次覆盖


def keyadd(p,x):
    all_keys = set(p.keys()).union(x.keys())
    result = OrderedDict()
    for key in all_keys:
        result[key] = p.get(key, 0) + x.get(key, 0)
    return result

if __name__ == '__main__':

    # genre_min_fitness_values_sum=[526, 440, 450, 437, 429 ,442, 421.2 ,422, 410 ,412 ,406 ,381.6 ,392 ,375, 367 ,377.2, 379.3 ,367 ,369, 372, 358, 357 ,358 ,361,367.5 ,358, 362 ,368 ,360.4 ,361 ,363, 358.6 ,373, 368.1, 368.4 ,360 ,368 ,375 ,366 ,356, 367, 362 ,358.6 ,366.6 ,356, 357.3, 357, 356, 354]
    # 所有运行的结果汇总
    run_fitness_history = []
    # 记录每一代的测试集最小适应度的累加值
    genre_min_fitness_values_sum = [0] * NGEN
    # 记录每棵树的叶子类型统计比例累加值
    leaf_ratio_result_sum = [OrderedDict() for _ in range(NUM_TREES)]
    # NUM_RUNS 次独立运行
    for _ in range(NUM_RUNS):
        # 每次独立运行后，得到每一代的测试集的最小适应度，以及每棵树的叶子比例统计
        genre_min_fitness_values, leaf_ratio_result = main_dual_tree()

        run_fitness_history.append(genre_min_fitness_values)
        genre_min_fitness_values_sum = [a + b for a, b in
                                        zip(genre_min_fitness_values, genre_min_fitness_values_sum)]
        leaf_ratio_result_sum = [keyadd(a, b) for a, b in zip(leaf_ratio_result, leaf_ratio_result_sum)]

    genre_min_fitness_values_sum = [a / NUM_RUNS for a in genre_min_fitness_values_sum]

    print(leaf_ratio_result_sum)
    print(run_fitness_history)

    # min_fitness_trend(genre_min_fitness_values_sum)



