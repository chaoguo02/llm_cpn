import operator
import random
import copy
import json
import os
from deap import gp
from collections import OrderedDict
from baselines.my_fifo import get_fifo_node_score, get_fifo_task_score
from core_function.pset_toolbox_settings import init_pset_toolbox
from core_function.running_process import eaSimple
from core_function.data_loader import createNode, createTaskFlows
from core_function.evaluate import work_processing
from core_function.update_time import computing_Task, computing_upload_time
from dirty_work.dirty_works import protect_div, count_leaf_types


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



def main_dual_tree(num_run):
    nodes = createNode(NUM_NODES,0)
    pre_generated_taskflows = []
    for i in range(NUM_TEST_SETS):
        taskflow = createTaskFlows(NUM_TASKFLOWS,1,i)
        pre_generated_taskflows.append(taskflow)

    pset, toolbox = init_pset_toolbox(TOURNAMENT_SIZE)

    population = toolbox.population(n=POP_SIZE)
    pop,genre_min_fitness_values,elite= eaSimple(population=population,
                                                       toolbox=toolbox,
                                                       nodes=nodes,
                                                       pre_generated_taskflows=pre_generated_taskflows,
                                                       num_TaskFlow=NUM_TASKFLOWS,
                                                       cxpb=CXPB,
                                                       mutpb=MUTPB,
                                                       ngen=NGEN,
                                                       elitism=ELITISM_NUM,
                                                       pset=pset,
                                                       num_run = num_run,
                                                       base_seed = BASE_SEED,
                                                       num_train_sets = NUM_TRAIN_SETS,
                                                       num_test_sets = NUM_TEST_SETS,
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



