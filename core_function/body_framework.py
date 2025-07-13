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
from dirty_work.dirty_works import count_leaf_types

def main_dual_tree(num_run, pre_generated_taskflows,num_nodes,tournament_size,
                   pop_size,num_taskflows,cxpb, mutpb,ngen,elitism_num,
                   base_seed,num_train_sets,num_test_sets):
    nodes = createNode(num_nodes)
    pset, toolbox = init_pset_toolbox(tournament_size)
    population = toolbox.population(n=pop_size)
    pop,genre_min_fitness_values,elite= eaSimple(population=population,
                                                       toolbox=toolbox,
                                                       nodes=nodes,
                                                       pre_generated_taskflows=pre_generated_taskflows,
                                                       num_taskflow=num_taskflows,
                                                       cxpb=cxpb,
                                                       mutpb=mutpb,
                                                       ngen=ngen,
                                                       elitism=elitism_num,
                                                       pset=pset,
                                                       num_run = num_run,
                                                       base_seed = base_seed,
                                                       num_train_sets = num_train_sets,
                                                       num_test_sets = num_test_sets,
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




