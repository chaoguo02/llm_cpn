from collections import OrderedDict

from core_function.data_loader import createTaskFlows
from core_function.body_framework import main_dual_tree


if __name__ == '__main__':

    #  统一参数设置
    NUM_NODES = 5  # 节点数
    NUM_TASKFLOWS = 12  # 任务流数
    POP_SIZE = 20  # 种群规模
    NGEN = 2  # 进化代数
    CXPB = 0.8  # 交叉概率
    MUTPB = 0.1  # 变异概率
    TOURNAMENT_SIZE = 1  # 锦标赛选择规模
    HEIHT_LIMIT = 6  # 最大树高
    NUM_TREES = 2  # 每个个体中树的数量
    NUM_RUNS = 1  # 重复运行次数（用于统计均值）
    ELITISM_NUM = int(POP_SIZE * 0.05)  # 精英个体的数量
    NUM_TEST_SETS = 1  # 测试集数量
    NUM_TRAIN_SETS = 1  # 训练集数量
    BASE_SEED = 10000  # 训练集我们从10000开始

    run_fitness_history = []
    # 记录每一代的测试集最小适应度的累加值
    avg_fitness_per_gen = [0] * NGEN
    # 记录每棵树的叶子类型统计比例累加值
    leaf_ratio_result_sum = [OrderedDict() for _ in range(NUM_TREES)]
    # NUM_RUNS 次独立运行
    pre_generated_taskflows = []
    for i in range(NUM_TEST_SETS):
        taskflow = createTaskFlows(NUM_TASKFLOWS, 1, i)
        pre_generated_taskflows.append(taskflow)
    for num_run in range(NUM_RUNS):
        # 每次独立运行后，得到每一代的测试集的最小适应度，以及每棵树的叶子比例统计
        min_fitness_per_gen, leaf_ratio_result = main_dual_tree(num_run, pre_generated_taskflows,NUM_NODES,TOURNAMENT_SIZE,
                                                                POP_SIZE,NUM_TASKFLOWS, CXPB, MUTPB,NGEN,ELITISM_NUM,
                                                                BASE_SEED,NUM_TRAIN_SETS,NUM_TEST_SETS)

        run_fitness_history.append(min_fitness_per_gen)
        avg_fitness_per_gen = [a + b for a, b in zip(min_fitness_per_gen, avg_fitness_per_gen)]

        leaf_ratio_result_sum = []
        for a, b in zip(leaf_ratio_result, leaf_ratio_result_sum):
            all_keys = set(a.keys()).union(b.keys())
            merged = OrderedDict()
            for key in all_keys:
                merged[key] = a.get(key, 0) + b.get(key, 0)
            leaf_ratio_result_sum.append(merged)

    avg_fitness_per_gen = [a / NUM_RUNS for a in avg_fitness_per_gen]