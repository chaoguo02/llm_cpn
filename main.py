from collections import OrderedDict
import argparse
from core_function.data_loader import createTaskFlows
from core_function.body_framework import main_dual_tree
from core_function.get_decode_functions import get_decode_functions

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', type=str, default='random',
                        choices=['gp', 'fifo','greedy','sjf','random','minmin','minmax'],
                        help='Specify which scheduling strategy to use: gp or fifo')
    return parser.parse_args()

if __name__ == '__main__':
    # 统一参数设置
    NUM_NODES = 10  # 节点数
    NUM_TASKFLOWS = 20  # 任务流数
    POP_SIZE = 20  # 种群规模
    NGEN = 0  # 进化代数
    CXPB = 0.84  # 交叉概率
    MUTPB = 0.11  # 变异概率
    TOURNAMENT_SIZE = 1  # 锦标赛选择规模
    HEIHT_LIMIT = 6  # 最大树高
    NUM_TREES = 2  # 每个个体中树的数量
    NUM_RUNS = 1  # 重复运行次数（用于统计均值）
    ELITISM_NUM = max(1, int(POP_SIZE * 0.05))  # 精英个体的数量
    NUM_TEST_SETS = 1  # 测试集数量
    NUM_TRAIN_SETS = 1  # 训练集数量
    BASE_SEED = 10000  # 训练集我们从10000开始

    args = parse_args()

    # 设置全局调度策略
    decode1, decode2 = get_decode_functions(args.strategy)
    run_fitness_history = []
    avg_fitness_per_gen = [0] * NGEN
    leaf_ratio_result_sum = [OrderedDict() for _ in range(NUM_TREES)]

    pre_generated_taskflows = [createTaskFlows(NUM_TASKFLOWS, 1, i) for i in range(NUM_TEST_SETS)]

    for num_run in range(NUM_RUNS):
        min_fitness_per_gen, leaf_ratio_result = main_dual_tree(
            num_run, pre_generated_taskflows,
            NUM_NODES, TOURNAMENT_SIZE,
            POP_SIZE, NUM_TASKFLOWS,
            CXPB, MUTPB, NGEN, ELITISM_NUM,
            BASE_SEED, NUM_TRAIN_SETS, NUM_TEST_SETS,
            decode1, decode2
        )

        run_fitness_history.append(min_fitness_per_gen)
        avg_fitness_per_gen = [a + b for a, b in zip(min_fitness_per_gen, avg_fitness_per_gen)]

        for i in range(NUM_TREES):
            a = leaf_ratio_result[i]
            b = leaf_ratio_result_sum[i]
            merged = OrderedDict()
            for key in set(a.keys()).union(b.keys()):
                merged[key] = a.get(key, 0) + b.get(key, 0)
            leaf_ratio_result_sum[i] = merged

    avg_fitness_per_gen = [a / NUM_RUNS for a in avg_fitness_per_gen]
