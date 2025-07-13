import copy
import json
import os

from core_function.evaluate import work_processing

#
# def record_best_individual_log(individual, pre_generated_taskflows, nodes, pset, generation_index, run_index,decode1,decode2, base_dir="best_ind_logs"):
#
#     test_taskflows_sample = copy.deepcopy(pre_generated_taskflows[0])
#     nodes_log = copy.deepcopy(nodes)
#
#     _, log_dict = work_processing(individual, test_taskflows_sample, nodes_log, pset,decode1,decode2, return_log=True)
#
#     log_dict["individual_expressions"] = [str(tree) for tree in individual]
#
#     run_dir = os.path.join(base_dir, f"run_{run_index}")
#     os.makedirs(run_dir, exist_ok=True)
#
#     log_path = os.path.join(run_dir, f"gen_{generation_index}_log.json")
#     with open(log_path, "w", encoding='utf-8') as f:
#         json.dump(log_dict, f, indent=2, ensure_ascii=False)

def record_best_individual_log(individual, pre_generated_taskflows, nodes, pset,
                                generation_index, run_index,
                                decode1, decode2, strategy_name,
                                base_dir="best_ind_logs"):

    test_taskflows_sample = copy.deepcopy(pre_generated_taskflows[0])
    nodes_log = copy.deepcopy(nodes)

    _, log_dict = work_processing(individual, test_taskflows_sample, nodes_log, pset,
                                  decode1=decode1, decode2=decode2, return_log=True)

    log_dict["individual_expressions"] = [str(tree) for tree in individual]
    log_dict["decode_strategy"] = strategy_name

    # 添加 strategy_name 作为一级子目录
    strategy_dir = os.path.join(base_dir, strategy_name)
    run_dir = os.path.join(strategy_dir, f"run_{run_index}")
    os.makedirs(run_dir, exist_ok=True)

    log_path = os.path.join(run_dir, f"gen_{generation_index}_log.json")
    with open(log_path, "w", encoding='utf-8') as f:
        json.dump(log_dict, f, indent=2, ensure_ascii=False)
