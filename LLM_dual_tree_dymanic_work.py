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

from llm_engine.llm_evolutionary_operators import llm_mutated_expressions, \
    llm_crossover_expressions
from utils.convert_tree2expression import tree_to_expression, expression_to_tree
from utils.openai_interface import OpenAIInterface

#  统一参数设置
NUM_NODES = 10                 # 节点数
NUM_TASKFLOWS = 20             # 任务流数
POP_SIZE = 20                  # 种群规模
NGEN = 2                       # 进化代数
CXPB = 0.9                     # 交叉概率
MUTPB = 0.1                    # 变异概率
TOURNAMENT_SIZE = 5           # 锦标赛选择规模
HEIHT_LIMIT = 8               # 最大树高
NUM_TREES = 2                 # 每个个体中树的数量
NUM_RUNS = 2                # 重复运行次数（用于统计均值）
EVAL_TEST_TASKFLOWS = 1       # 测试时的任务流数
ELITISM_NUM = int(POP_SIZE * 0.05)  # 精英个体的数量
LLM_INTERFACE = OpenAIInterface()
NUM_TEST_SETS = 1             # 测试集数量

def initIndividual(container, func, pset, size):
    return container(gp.PrimitiveTree(func(pset[i])) for i in range(size))

def cxOnePointListOfTrees(ind1, ind2, pset=None, llm_interface=None):
    assert pset is not None, "❌ `pset` 不能为 None！"
    HEIGHT_LIMIT = 8
    print("🚀 进入 cxOnePointListOfTrees 函数")

    try:
        for idx, (tree1, tree2) in enumerate(zip(ind1, ind2)):
            print(f"\n===== 🧬 正在处理第 {idx} 棵子树 =====")
            print("🔵 原始 tree1:", str(tree1))
            print("🔵 原始 tree2:", str(tree2))
            print(f"📦 当前子树类型: {type(tree1)} / {type(ind1[idx])}")

            if idx >= len(pset):
                raise IndexError(f"❌ idx {idx} 超过了 pset 的定义范围")

            expr1 = tree_to_expression(tree1)
            expr2 = tree_to_expression(tree2)
            print(f"🔍 转换后的表达式:\n  expr1: {expr1}\n  expr2: {expr2}")

            print("⚙️ 调用 llm_crossover_expressions 执行交叉...")
            if idx == 0:
                new_expressions = llm_crossover_expressions(llm_interface, [expr1, expr2], 0)
                pset_idx = 0
            elif idx == 1:
                new_expressions = llm_crossover_expressions(llm_interface, [expr1, expr2], 1)
                pset_idx = 1
            else:
                raise ValueError(f"❌ 不支持的 idx 值: {idx}")

            print("🧪 LLM 生成的新表达式:", new_expressions)

            # 构建新树并加异常捕获
            try:
                new_tree1 = gp.PrimitiveTree.from_string(expression_to_tree(new_expressions[0]), pset[pset_idx])
            except Exception as e:
                print(f"❌ new_tree1 构建失败: {e}，使用原始 tree1")
                new_tree1 = tree1

            try:
                new_tree2 = gp.PrimitiveTree.from_string(expression_to_tree(new_expressions[1]), pset[pset_idx])
            except Exception as e:
                print(f"❌ new_tree2 构建失败: {e}，使用原始 tree2")
                new_tree2 = tree2

            print(f"🌲 新树高度: tree1: {new_tree1.height}, tree2: {new_tree2.height}")

            # 判断是否超高并构造新 individual
            if new_tree1.height > HEIGHT_LIMIT:
                print(f"⚠️ new_tree1 超出高度限制，使用原始 individual1")
                new_individual1 = ind1[idx]
            else:
                new_individual1 = type(ind1[idx])(new_tree1)

            if new_tree2.height > HEIGHT_LIMIT:
                print(f"⚠️ new_tree2 超出高度限制，使用原始 individual2")
                new_individual2 = ind2[idx]
            else:
                new_individual2 = type(ind2[idx])(new_tree2)

            # 替换原个体
            ind1[idx], ind2[idx] = new_individual1, new_individual2

        print("✅ 所有子树交叉完成，返回 ind1, ind2")
        return ind1, ind2

    except Exception as e:
        print("❌ 交叉过程中发生异常：", e)
        raise e

def mutUniformListOfTrees(individual, pset=None, llm_interface=None):
    assert pset is not None, "❌ `pset` 不能为 None！"
    HEIGHT_LIMIT = 8
    try:
        for idx, tree in enumerate(individual):
            if idx >= len(pset):
                raise IndexError(f"idx {idx} 超过了pset的定义范围")
            print(f"===== 变异处理第 {idx} 棵子树 =====")
            print("🟢 原始 tree (str):", str(tree))

            expr = tree_to_expression(tree)
            print("🔍 转换为表达式:", expr)

            # 调用 LLM 执行变异
            new_expression = llm_mutated_expressions(llm_interface, expr, idx)
            print("🎯 LLM 生成的新表达式:", new_expression)

            # 尝试构造新树
            try:
                new_tree = gp.PrimitiveTree.from_string(expression_to_tree(new_expression), pset[idx])
            except Exception as e:
                print(f"❌ 构建 new_tree 失败: {e}，使用原始 tree")
                new_tree = tree

            # 判断高度限制
            if new_tree.height > HEIGHT_LIMIT:
                print(f"⚠️ new_tree 超出高度限制，使用原始 tree")
                new_individual = individual[idx]
            else:
                new_individual = type(individual[idx])(new_tree)

            # 替换个体中对应子树
            individual[idx] = new_individual

        print("✅ 所有子树变异完成，返回 individual")
        return (individual,)

    except Exception as e:
        print(f"❌ 变异过程中发生异常：{e}，返回原始 individual")
        return (individual,)

def varAnd(population, toolbox, cxpb, mutpb, pset):
    offspring = [toolbox.clone(ind) for ind in population]
    evlpb=cxpb/(cxpb+mutpb)
    if random.random() < evlpb:
        for i in range(1, len(offspring), 2):
            offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1],
                                                          offspring[i], pset,LLM_INTERFACE)
            del offspring[i - 1].fitness.values, offspring[i].fitness.values

    else:
        for i in range(len(offspring)):
            offspring[i], = toolbox.mutate(offspring[i],pset,LLM_INTERFACE)
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

def evaluate_offspring_in_parallel(offspring, taskflows, nodes, pset, work_function):

    pool = multiprocessing.Pool()
    results = []

    for ind in offspring:
        tasks_copy = copy.deepcopy(taskflows)
        nodes_copy = copy.deepcopy(nodes)
        result = pool.apply_async(work_function, (ind, tasks_copy, nodes_copy, pset,))
        results.append(result)

    pool.close()
    pool.join()

    fitnesses = [res.get() for res in results]
    return fitnesses

def evaluate_on_testSets(individual, nodes, num_TaskFlow, pset):
    results = []
    for i in range(NUM_TEST_SETS):
        nodes_copy = copy.deepcopy(nodes)
        taskflows = createTaskFlows(num_TaskFlow, genre=1, seed=i)
        fitness = work_processing(individual, taskflows, nodes_copy, pset)
        results.append(fitness[0])
    return results

def eaSimple(population, toolbox,taskflows,nodes,num_TaskFlow,num_nodes, cxpb, mutpb, ngen, elitism,pset,llm_interface,min_fitness_values=[],genre_min_fitness_values=[],stats=None,
             halloffame=None, verbose=__debug__, ):
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # 筛选出未适应度评估的个体
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    # 适应度评估(训练集)
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
    # 适应度评估（验证集）
    value_list = evaluate_on_testSets(sorted_elite[0], nodes, num_TaskFlow, pset)
    genre_min_fitness_values.append(sum(value_list) / NUM_TEST_SETS)

    # 开启进化过程
    for gen in range(1, ngen + 1):
        taskflows = createTaskFlows(num_TaskFlow,0,gen)

        # Select the next generation individuals
        offspring = toolbox.select(population, len(population)-elitism) # 这行代码使用选择操作从当前种群中选择与原种群相同数量的个体，这些个体构成新的后代。,只是选择的过程为3选一

        offspring = varAnd(offspring, toolbox, cxpb, mutpb, pset)

        offspring[:] = sorted_elite + offspring

        # 适应度评估（训练集）
        fitnesses = evaluate_offspring_in_parallel(offspring,taskflows,nodes,pset,work_processing)

        for ind, fit in zip(offspring, fitnesses):
            ind.fitness.values = fit   #将这些计算出的适应度值赋给对应的个体，使得它们的适应度属性变为有效。

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        min_fitness_values.append(record["fitness"]["min"]) #我加的
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)

        if verbose:
            print(logbook.stream)

        sorted_elite = sortPopulation(toolbox, population)[:elitism]

        # 适应度评估（测试集）
        value_list = evaluate_on_testSets(sorted_elite[0], nodes, num_TaskFlow, pset)
        genre_min_fitness_values.append(sum(value_list) / NUM_TEST_SETS)

    return population, logbook ,genre_min_fitness_values,sorted_elite[0]

def is_number(string):#创建一个函数 is_number，用于检查一个字符串是否可以转换为浮点数。如果可以，则返回 True，否则返回 False。
    try:
        float(string)
        return True
    except ValueError:
        return False

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
    taskflows=[TaskFlow(id,arrival_time,genre,id + (seed) * num_TaskFlow) for id, arrival_time in zip(range(num_TaskFlow), arrival_times)]
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
        for event in queue2:
            if event[1]==min_time:
                task_queue2.append(event[0])
    elif len(queue1)==0  and len(queue2)!=0:
        min_time = queue2[0][1]
        for event in queue2:
            if event[1]==min_time:
                task_queue2.append(event[0])
    elif len(queue1)!=0  and len(queue2)==0:
        min_time = queue1[0][1]
        for event in queue1:
            if event[1]==min_time:
                task_queue1.append(event[0])
    return queue1,queue2,task_queue1,task_queue2

#decode1函数中需要增加一个约束条件，判断节点的ram，gpu，cpu够不够用，要在够用的节点中选择
def decode1(individual, task, nodes, taskflows,pset):
    heuristic_1 = gp.compile(expr=individual[0], pset=pset[0])
    scores = []
    # k=task.taskflow_id #任务所在任务流，在多任务流中的位置
    for node in nodes:
        heuristic_score = heuristic_1(computing_Task(task,node),task.present_time-task.arrivetime,
                                      len(task.descendant),computing_upload_time(task,node))
        scores.append((node, heuristic_score))
    best_node = max(scores, key=lambda x: x[1])[0]
    return best_node #找到节点本身

def decode2(individual, node, taskflows,nodes, pset):
    heuristic_2 = gp.compile(expr=individual[1], pset=pset[1])
    scores = []
    for task in node.waiting_queue:
        k = task.taskflow_id#任务所在任务流，在多任务流中的位置
        heuristic_score = heuristic_2(computing_Task(task,node),task.present_time-task.arrivetime,
                                      len(task.descendant),taskflows[k].find_descendant_avg_time(taskflows,task,nodes),
                                      0.1*computing_upload_time(task,node)) #任务给后继传递消息的时间取上传时间的0.1
        scores.append((task, heuristic_score))
    #print(scores)
    best_task = max(scores, key=lambda x : x[1])[0]
    return best_task #返回找到的任务本身

def work_processing(individual, taskflows, nodes, pset):
    def sanitize_task_id(task):
        return getattr(task, "global_id", f"Task {task.id}")

    def sanitize_node_id(node):
        return f"N{node.id}({node.node_type})"

    queue1 = []  # 未执行任务队列
    queue2 = []  # 正在执行的任务队列
    present_time = 0
    present_time_update(present_time, taskflows)

    print("🚀 开始任务流调度模拟...")
    for taskflow in taskflows:
        tasks = taskflow.find_predecessor_is_zero()
        for task in tasks:
            queue1.append((task, task.arrivetime))
            print(f"🕓 任务 Task {task.task_id} 到达时间 {task.arrivetime} 已加入 queue1")

    while len(queue2) != 0 or len(queue1) != 0:
        queue1, queue2, task_queue1, task_queue2 = find_earlist_time(queue1, queue2)

        if len(task_queue1) != 0:
            present_time_update(present_time=queue1[0][1], taskflows=taskflows)
            print(f"\n📍 时间推进至 {queue1[0][1]}，调度以下待执行任务：")

            for task in task_queue1:
                node = decode1(individual, task, nodes, taskflows, pset)
                task.node = node
                print(f"🔍 Task {task.task_id} 分配给 Node {node.node_id}")

                if task.present_time >= node.begin_idle_time:
                    computing_Task_time = computing_Task(task, node) + computing_upload_time(task, node)
                    endtime = task.present_time + computing_Task_time
                    task.endtime = endtime
                    queue2.append((task, endtime))
                    node.begin_idle_time = endtime
                    print(f"✅ Task {task.task_id} 开始执行，预计完成于 {endtime}")
                else:
                    node.waiting_queue.append(task)
                    print(f"⏳ Task {task.task_id} 加入 Node {node.node_id} 的等待队列")
                queue1 = [item for item in queue1 if item[0] != task]

        if len(task_queue2) != 0:
            print(f"\n🏁 时间推进至 {queue2[0][1]}，以下任务完成：")
            for finish_event in task_queue2:
                finish_event.finish = True
                print(f"✅ Task {finish_event.task_id} 执行完成")

            present_time_update(present_time=queue2[0][1], taskflows=taskflows)

            for task in task_queue2:
                i = task.taskflow_id
                TaskFlow = taskflows[i]

                if len(task.descendant) != 0:
                    id = TaskFlow.tasks.index(task)
                    descendant_tasks = []
                    for b in task.descendant:
                        TaskFlow.tasks[b].predecessor.remove(id)
                        if len(TaskFlow.tasks[b].predecessor) == 0:
                            descendant_tasks.append(TaskFlow.tasks[b])

                    for descendant_task in descendant_tasks:
                        queue1.append((descendant_task, descendant_task.present_time))
                        print(f"🔄 后继任务 Task {descendant_task.task_id} 前驱已完成，加入 queue1")

                else:
                    TaskFlow.finish_time = max(TaskFlow.finish_time, queue2[0][1])
                    print(f"🎯 TaskFlow {i} 可能完成，当前记录完成时间为 {TaskFlow.finish_time}")

                if len(task.node.waiting_queue) != 0:
                    next_task = decode2(individual, task.node, taskflows, nodes, pset)
                    task.node.waiting_queue.remove(next_task)
                    computing_Task_time = computing_Task(next_task, task.node) + \
                                          computing_upload_time(next_task, task.node) + \
                                          0.1 * computing_upload_time(task, task.node)
                    next_task.endtime = next_task.present_time + computing_Task_time
                    queue2.append((next_task, next_task.endtime))
                    task.node.begin_idle_time = next_task.endtime
                    print(f"📤 Node {task.node.node_id} 从等待队列中调度 Task {next_task.task_id} 执行，完成于 {next_task.endtime}")
                queue2 = [item for item in queue2 if item[0] != task]

    print("\n📊 所有任务流调度结束，统计后10个任务流平均完成时间：")
    sum_time = 0
    taskflows_number = 0
    for j in taskflows[10:]:
        taskflows_number += 1
        print(f"🧾 TaskFlow 完成时间：{j.finish_time}，开始时间：{j.all_arrive_time}")
        sum_time += (j.finish_time - j.all_arrive_time)

    avg_time = sum_time / taskflows_number if taskflows_number else 0
    print(f"\n📈 平均完成时间（后10个任务流）: {avg_time}")
    return (avg_time,)


# def work_processing(individual,taskflows,nodes,pset): #taskflows本体和附属属性有没有改变？※等号传的都是地址
#     queue1=[]               # 未执行任务队列
#     queue2=[]               # 正在执行的任务队列
#     present_time=0
#     present_time_update(present_time, taskflows)
#
#
#     """
#     queue中元素为元组 （任务本身，任务的开始时间或任务的结束时间）
#     """
#
#     print("🚀 开始任务流调度模拟...")
#     for taskflow in taskflows:
#         tasks = taskflow.find_predecessor_is_zero()
#         for task in tasks:
#             queue1.append((task, task.arrivetime)) #元组
#             print(f"🕓 任务 Task {task.task_id} 到达时间 {task.arrivetime} 已加入 queue1")
#
#     while(len(queue2) != 0 or len(queue1) != 0):
#         # 找到最早发生的事件
#         queue1, queue2, task_queue1, task_queue2 = find_earlist_time(queue1,queue2)
#
#         if len(task_queue1)!=0:#在queue1和queue2中时间最小的任务在queue1
#             present_time_update(present_time=queue1[0][1],taskflows=taskflows)#更新所有queue2和queue1中任务和所有node中等待队列中任务的当前时间 和预处理任务流中所有没上queue1的任务当前时间
#
#             # 将task_queue1中的任务进行调度
#             for task in task_queue1:
#                 node=decode1(individual, task, nodes, taskflows, pset)#为所有满足要求的任务选择节点,decode1函数中需要增加一个约束条件，判断节点的ram，gpu，cpu够不够用，要在够用的节点中选择
#                 task.node=node
#                 if task.present_time>=node.begin_idle_time:#task的当前时间大于等于节点的空闲时间
#                     computing_Task_time=computing_Task(task,node)+computing_upload_time(task,node)
#                     endtime=task.present_time+computing_Task_time#task完成时间=task当前时间+task计算时间 round(a, 1)
#                     task.endtime=endtime
#                     queue2.append((task,endtime))#添加元组
#                     node.begin_idle_time=endtime#node的空闲时间修改为task完成时间
#                 else:
#                     node.waiting_queue.append(task)
#                 queue1 = [item for item in queue1 if item[0] != task]#在queue1中删除任务
#
#         if len(task_queue2)!=0 :#在queue1和queue2中时间最小的任务在queue2
#             for finish_event in task_queue2:#标记任务已完成
#                 finish_event.finish=True
#             present_time_update(present_time=queue2[0][1],taskflows=taskflows)#更新所有queue2和queue1中任务和所有node中等待队列中任务的当前时间 和预处理任务流中所有没上queue1的任务当前时间(除了已完成的任务)
#
#
#             for task in task_queue2:
#                 i=task.taskflow_id#位置
#                 TaskFlow=taskflows[i]#需要找到taskflow在taskflows中的位置
#                 if len(task.descendant)!=0:#这个任务有后继节点
#                     a=task.descendant #task的所有后继下标
#                     id=TaskFlow.tasks.index(task) #task在tasks中的下标
#                     descendant_tasks=[] #在task的后继中找到所有前置为0的task
#                     for b in a:
#                         TaskFlow.tasks[b].predecessor.remove(id)                              #所有任务删除这个前继，向后继传递信息
#                         if len(TaskFlow.tasks[b].predecessor)==0:#后继任务没有了所有的前继任务
#                             descendant_tasks.append(TaskFlow.tasks[b])
#                     for descendant_task in descendant_tasks:
#                         queue1.append((descendant_task,descendant_task.present_time))#添加元组#该任务上queue1,带着当前时间组成的元组
#
#                 else:#没有后继
#                     TaskFlow.finish_time=max(TaskFlow.finish_time,queue2[0][1])#更新任务流的完成时间 取最大值
#                 if len(task.node.waiting_queue)!=0:#任务所在节点的等待队列不为空
#                     next_task=decode2(individual, task.node, taskflows,nodes, pset)#任务所在节点在其所在队列中选择一个任务
#                     task.node.waiting_queue.remove(next_task)
#                     computing_Task_time = computing_Task(next_task,task.node)+computing_upload_time(next_task,task.node)+0.1*computing_upload_time(task,task.node) #包括上个任务给后继传递消息的时间
#                     next_task.endtime=next_task.present_time+computing_Task_time#task完成时间=task当前时间+task计算时间
#                     queue2.append((next_task,next_task.endtime))#添加元组
#
#                     task.node.begin_idle_time=next_task.endtime#node的空闲时间修改为task完成时间 ,这里需要考虑是否修改了node的本体值
#                 queue2 = [item for item in queue2 if item[0] != task]#在queue1中删除任务#在queue2中删除任务
#
#     sum_time=0
#     taskflows_number=0
#     for j in taskflows[10:]:#遍历非前10个任务流算出任务流的平均完成时间
#         taskflows_number+=1
#         #print(j.finish_time,j.all_arrive_time)
#         sum_time+=(j.finish_time-j.all_arrive_time)
#     return (sum_time/taskflows_number,)

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
    toolbox.register("mutate", mutUniformListOfTrees)
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
                                                       num_TaskFlow=NUM_TASKFLOWS,
                                                       num_nodes=NUM_NODES,
                                                       cxpb=CXPB,
                                                       mutpb=MUTPB,
                                                       ngen=NGEN,
                                                       elitism=ELITISM_NUM,
                                                       pset=pset,
                                                       llm_interface=LLM_INTERFACE,
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

def keyadd(p,x):
    all_keys = set(p.keys()).union(x.keys())
    result = OrderedDict()
    for key in all_keys:
        result[key] = p.get(key, 0) + x.get(key, 0)
    return result

if __name__ == '__main__':

    #genre_min_fitness_values_sum=[526, 440, 450, 437, 429 ,442, 421.2 ,422, 410 ,412 ,406 ,381.6 ,392 ,375, 367 ,377.2, 379.3 ,367 ,369, 372, 358, 357 ,358 ,361,367.5 ,358, 362 ,368 ,360.4 ,361 ,363, 358.6 ,373, 368.1, 368.4 ,360 ,368 ,375 ,366 ,356, 367, 362 ,358.6 ,366.6 ,356, 357.3, 357, 356, 354]
    # 所有运行的结果汇总
    run_fitness_history = []
    # 记录每一代的测试集最小适应度的累加值
    genre_min_fitness_values_sum = [0] * NGEN
    # 记录每棵树的叶子类型统计比例累加值
    leaf_ratio_result_sum = [OrderedDict() for _ in range(NUM_TREES)]
    # NUM_RUNS 次独立运行
    for _ in range(NUM_RUNS):
        # 每次独立运行后，得到每一代的测试集的最小适应度，以及每棵树的叶子比例统计
        genre_min_fitness_values,leaf_ratio_result=main_dual_tree()

        run_fitness_history.append(genre_min_fitness_values)
        genre_min_fitness_values_sum = [a + b for a, b in zip(genre_min_fitness_values, genre_min_fitness_values_sum)]
        leaf_ratio_result_sum=[keyadd(a,b) for a,b in zip(leaf_ratio_result,leaf_ratio_result_sum)]

    genre_min_fitness_values_sum=[a / NUM_RUNS  for a in genre_min_fitness_values_sum]


    print(leaf_ratio_result_sum) 
    print(run_fitness_history)

    # min_fitness_trend(genre_min_fitness_values_sum)



