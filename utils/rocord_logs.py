import copy
import json
import os

from core_function.evaluate import work_processing


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
