import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns
import os
import json
from glob import glob

base_dir = "/root/vllm/benchmarks/result"
replace_name = {
    "fcfs": "FCFS",
    # "las": "LAS",
    # "srjf": "SRJF",
    "opt": "L2R",
    "sjmlfq": "MLFQ",
    "tfittradeoff": "Ours",
    "sjf": "SJF",
}


def generate_dir_names(base_dir, dates, counters):
    """生成目录路径列表"""
    dir_names = []
    for date, counter_list in zip(dates, counters):
        dir_names.append([os.path.join(base_dir, date, str(counter)) for counter in counter_list])
    return dir_names


def get_seq_result(model_id, dirname, scheduling_policy, parallel_type, all_results):
    all_file_names = os.listdir(dirname)
    all_result_files = []
    for file_name in all_file_names:
        if (
            model_id in file_name
            and scheduling_policy in file_name
            and parallel_type in file_name
            and "seq_level" in file_name
        ):
            all_result_files.append(os.path.join(dirname, file_name))
    for result_file in all_result_files:
        qps = result_file.split("/")[-1].split(".")[0]
        result = pd.read_csv(result_file)
        result.loc[:, "request_rate"] = int(qps)
        result.loc[:, "scheduling_policy"] = scheduling_policy
        result.loc[:, "parallel_type"] = parallel_type
        result.loc[:, "model_id"] = model_id
        result.loc[:, "waiting_times"] = result.loc[:, "waiting_times"].apply(lambda x: eval(x))
        result.replace(replace_name, inplace=True)
        all_results.append(result)


def get_system_result(model_id, dirname, scheduling_policy, parallel_type, all_results):
    all_file_names = os.listdir(dirname)
    all_result_files = []
    for file_name in all_file_names:
        if (
            model_id in file_name
            and scheduling_policy in file_name
            and parallel_type in file_name
            and "system_level" in file_name
        ):
            all_result_files.append(os.path.join(dirname, file_name))
    for result_file in all_result_files:
        qps = result_file.split("/")[-1].split(".")[0]
        result = pd.read_csv(result_file)
        result.loc[:, "request_rate"] = int(qps)
        result.loc[:, "scheduling_policy"] = scheduling_policy
        result.loc[:, "parallel_type"] = parallel_type
        result.loc[:, "model_id"] = model_id
        result.loc[:, "avg_decode_token_nums"] = result.loc[:, "decode_token_num"] / result["running_seq_nums"]
        result.loc[:, "gpu_memory_occupy"] = (
            result.loc[:, "gpu_memory_occupy"] - result.loc[:, "running_seq_nums"] / 3684
        )
        result.loc[:, "gpu_memory_occupy"] = result.loc[:, "gpu_memory_occupy"] / 16
        result.replace(replace_name, inplace=True)
        all_results.append(result)


def seq_result_13b():
    data_types = ["/root/vllm/benchmarks/result/fixed_result/share_gpt/llama2-13b/microscope"]
    methods = [["fcfs", "l2r", "mlfq", "ours", "sjf"]]
    scheduling_policies = list(replace_name.keys())
    models = ["Llama-2-13b"]
    parallel_types = ["single"]
    all_results = []

    seq_level_result_dir_names = generate_dir_names(base_dir, data_types, methods)

    for seq_level_result_dir_name in seq_level_result_dir_names:
        for model in models:
            for result_dir_name in seq_level_result_dir_name:
                for scheduling_policy in scheduling_policies:
                    for parallel_type in parallel_types:
                        get_seq_result(model, result_dir_name, scheduling_policy, parallel_type, all_results)
    total_seq_results = pd.concat(all_results)
    total_seq_results.loc[:, "max_waiting_time"] = total_seq_results.loc[:, "waiting_times"].apply(
        lambda x: np.max(x[1:])
    )
    total_seq_results.loc[:, "avg_waiting_times"] = total_seq_results.loc[:, "waiting_times"].apply(
        lambda x: np.mean(x[1:])
    )
    total_seq_results.to_csv(f"{data_types[0]}/total_seq_results.csv", index=False)


def system_result_13b():
    all_system_results = []
    data_types = ["/root/vllm/benchmarks/result/fixed_result/share_gpt/llama2-13b/microscope"]
    methods = [["fcfs", "l2r", "mlfq", "ours", "sjf"]]
    e2e_result_dir_names = generate_dir_names(base_dir, data_types, methods)
    models = ["Llama-2-13b"]
    parallel_types = ["single"]
    scheduling_policies = list(replace_name.keys())
    for e2e_result_dir_name, counter in zip(e2e_result_dir_names, methods):
        for i in range(len(models)):
            model = models[i]
            parallel_type = parallel_types[i]
            for result_dir_name in e2e_result_dir_name:
                for scheduling_policy in scheduling_policies:
                    get_system_result(model, result_dir_name, scheduling_policy, parallel_type, all_system_results)
    total_system_results = pd.concat(all_system_results)
    total_system_results.sort_values(by=["request_rate", "scheduling_policy"], inplace=True)
    total_system_results.to_csv(f"{data_types[0]}/total_system_results.csv", index=False)

seq_result_13b()
system_result_13b()
