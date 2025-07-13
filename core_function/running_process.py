import numpy as np
from core_function.data_loader import createTaskFlows
from core_function.evaluate import evaluate_offspring_in_parallel, work_processing, evaluate_on_testSets
from core_function.operators import sortPopulation, varAnd
from utils.rocord_logs import record_best_individual_log


def eaSimple(population, toolbox, nodes, pre_generated_taskflows, num_taskflow,
             cxpb, mutpb, ngen, elitism, pset, num_run,base_seed,num_train_sets,num_test_sets, min_fitness_per_gen=None):

    if min_fitness_per_gen is None:
        min_fitness_per_gen = []

    # 初始种群评估（这里也是要对训练集进行改造的）
    initial_inds = [ind for ind in population if not ind.fitness.valid]
    initial_taskflows_list = [
        createTaskFlows(num_taskflow, 0, base_seed + i)
        for i in range(num_train_sets)
    ]
    fitnesses_list = [
        evaluate_offspring_in_parallel(initial_inds, taskflows, nodes, pset, work_processing)
        for taskflows in initial_taskflows_list
    ]

    fitnesses_avg = [
        (np.mean([fitnesses_list[run][i][0] for run in range(num_train_sets)]),)
        for i in range(len(initial_inds))
    ]

    for ind, fit in zip(initial_inds, fitnesses_avg):
        ind.fitness.values = fit

    elite_inds = sortPopulation(toolbox, population)[:elitism]

    # 测试集评估最优个体
    test_fitnesses = evaluate_on_testSets(elite_inds[0], nodes, pset, pre_generated_taskflows)
    min_fitness_per_gen.append(sum(test_fitnesses) / num_test_sets)

    # 保存第0代最优个体调度日志和表达式
    record_best_individual_log(elite_inds[0],pre_generated_taskflows,nodes,pset,0,num_run)

    for gen in range(1, ngen + 1):

        # 每一轮训练的时候改为两次仿真
        # train_taskflows = createTaskFlows(num_TaskFlow, 0, gen)
        train_taskflows_list = [
            createTaskFlows(num_taskflow, 0, base_seed + gen * num_train_sets + i)
            for i in range(num_train_sets)
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
            avg_fit = np.mean([fitnesses_list[run][i][0] for run in range(num_train_sets)])
            fitnesses_avg.append((avg_fit,))

        # 更新个体适应度
        for ind, fit in zip(offspring_inds, fitnesses_avg):
            ind.fitness.values = fit

        population[:] = offspring_inds

        elite_inds = sortPopulation(toolbox, population)[:elitism]

        # 测试集评估最优个体
        test_fitnesses = evaluate_on_testSets(elite_inds[0], nodes, pset, pre_generated_taskflows)
        min_fitness_per_gen.append(sum(test_fitnesses) / num_test_sets)

        # 保存当前代最优个体调度日志和表达式
        record_best_individual_log(elite_inds[0],pre_generated_taskflows,nodes,pset,gen,num_run)

    return population, min_fitness_per_gen, elite_inds[0]
