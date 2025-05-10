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


def get_metric_ratio(df, metric_min_result):
    min_result = metric_min_result[metric_min_result["Metric"] == df["Metric"].values[0]]["Value"].values[0]
    df["Ratio"] = round(df["Value"] / min_result, 2)
    return df


def get_metric_ratio_result(df, columns):
    df[columns] = round(df[columns] / 1, 2)
    return df


def generate_dir_names(base_dir, dates, counters):
    """生成目录路径列表"""
    dir_names = []
    for date, counter_list in zip(dates, counters):
        dir_names.extend([os.path.join(base_dir, date, str(counter)) for counter in counter_list])
    return dir_names


def process_file(file_path, schedule_policies, qps, detailed):
    """处理单个文件并返回策略和 DataFrame"""
    if not file_path.endswith(".csv") or qps not in file_path:
        return None, None

    if not detailed and "_detailed" not in file_path:
        df = pd.read_csv(file_path)
        df["Cache Efficiency"] = df["Running"] / df["GPU KV cache usage"]
    elif detailed and "_detailed" in file_path:
        df = pd.read_csv(file_path)
    else:
        return None, None

    for policy in schedule_policies:
        if policy in file_path:
            return policy.upper(), df
    return None, None


def load_execute_results(dir_names, schedule_policies, qps, detailed):
    """加载执行结果并返回字典"""
    execute_result_dfs = {}
    for dir_name in dir_names:
        for file_name in os.listdir(dir_name):
            file_path = os.path.join(dir_name, file_name)
            policy, df = process_file(file_path, schedule_policies, qps, detailed)
            if policy and df is not None:
                execute_result_dfs[policy] = df
    return execute_result_dfs


def get_result(e2e_result_dir_names):
    json_files = []
    for dir_name in e2e_result_dir_names:
        json_files.extend(glob(os.path.join(dir_name, "*.json")))
    e2e_result_dfs = {}
    for file_path in json_files:
        if "prompt" in file_path:
            continue
        with open(file_path, "r") as f:
            data = json.load(f)
        # 创建 DataFrame 并删除不需要的列
        e2e_result_df = pd.DataFrame(data).drop(
            columns=[
                "date",
                "backend",
                "tokenizer_id",
                "best_of",
                "use_beam_search",
            ]
        )
        if e2e_result_df["parallel_type"][0] == "tp" and "tp" not in file_path:
            continue

        e2e_result_df["avg_token_latency"] = e2e_result_df["latencies"] / e2e_result_df["output_lens"]
        # 替换 DataFrame 中的值
        e2e_result_df.replace(replace_name, inplace=True)

        # 将处理后的 DataFrame 存入字典
        file_name = os.path.basename(file_path)
        e2e_result_dfs[file_name] = e2e_result_df
    return e2e_result_dfs


def extract_e2e_results(e2e_result_dfs):
    """从多个 DataFrame 中提取结果并返回一个聚合后的 DataFrame"""
    e2e_result = {
        "scheduling_policies": [],
        "request_throughput": [],
        "output_throughput": [],
        "request_rates": [],
        "parallel_types": [],
        "model_ids": [],
    }

    for df_name, df in e2e_result_dfs.items():
        e2e_result["scheduling_policies"].append(df["scheduler_policy"].iloc[0])
        e2e_result["request_throughput"].append(df["request_throughput"].mean())
        e2e_result["output_throughput"].append(df["output_throughput"].mean())
        e2e_result["request_rates"].append(df["request_rate"].iloc[0])
        e2e_result["parallel_types"].append(df["parallel_type"].iloc[0])
        e2e_result["model_ids"].append(df["model_id"].iloc[0])

    result_df = pd.DataFrame(e2e_result)
    result_df = result_df[result_df["scheduling_policies"] != "las"]
    # 按策略和请求率分组并取最大值
    result_df = (
        result_df.groupby(["scheduling_policies", "request_rates", "parallel_types", "model_ids"]).mean().reset_index()
    )

    # 计算吞吐量比率
    result_df = result_df.groupby(["request_rates", "parallel_types", "model_ids"], group_keys=False).apply(
        lambda row: get_metric_ratio_result(row, "output_throughput")
    )
    result_df = result_df.groupby(["request_rates", "parallel_types", "model_ids"], group_keys=False).apply(
        lambda row: get_metric_ratio_result(row, "request_throughput")
    )

    return result_df


def process_selected_result(selected_result, selected_columns):
    """处理 selected_result 并返回长格式的 DataFrame"""
    # 转换为 DataFrame
    result_df = pd.DataFrame(selected_result)

    # 按策略和请求率分组并计算均值
    result_df = (
        result_df.groupby(["scheduler_policy", "swap_policy", "request_rate", "parallel_type", "model_ids"])
        .mean()
        .reset_index()
    )

    # 选择需要的列并转换为长格式
    long_df = result_df[
        ["scheduler_policy", "swap_policy", "request_rate", "parallel_type", "model_ids"] + selected_columns
    ].melt(
        id_vars=["scheduler_policy", "swap_policy", "request_rate", "parallel_type", "model_ids"],
        value_vars=selected_columns,
        var_name="Metric",
        value_name="Value",
    )

    # 计算指标比率（假设 get_metric_ratio 是一个自定义函数）
    metric_min_result = long_df.groupby(["Metric", "parallel_type", "model_ids"], group_keys=False).min().reset_index()
    long_df = (
        long_df.groupby(["Metric", "request_rate", "parallel_type", "model_ids"], group_keys=False)
        .apply(lambda row: get_metric_ratio(row, metric_min_result))
        .reset_index()
    )

    # 拆分 Metric 列
    long_df[["metric_name", "metric_type"]] = long_df["Metric"].apply(
        lambda row: pd.Series([row.split("_", 2)[0].capitalize(), row.split("_", 2)[1].upper()])
    )

    return long_df


def extract_selected_results(ttft_tpot_result_dfs):
    """从多个 DataFrame 中提取特定列的数据"""
    selected_result = {
        "scheduler_policy": [],
        "swap_policy": [],
        "request_rate": [],
        "parallel_type": [],
        "model_ids": [],
    }
    selected_columns = [
        "mean_ttft_ms",
        "median_ttft_ms",
        "p99_ttft_ms",
        "mean_tpot_ms",
        "median_tpot_ms",
        "p99_tpot_ms",
        "mean_itl_ms",
        "median_itl_ms",
        "p99_itl_ms",
        "mean_lat_ms",
        "median_lat_ms",
        "p99_lat_ms",
        "avg_token_latency",
    ]

    # 初始化 selected_columns 的值为空列表
    for column in selected_columns:
        selected_result[column] = []

    # 遍历每个 DataFrame 并提取数据
    for df_name, df in ttft_tpot_result_dfs.items():
        # 提取策略和请求率
        selected_result["scheduler_policy"].append(df["scheduler_policy"].iloc[0])
        selected_result["swap_policy"].append(df["swap_policy"].iloc[0])
        selected_result["request_rate"].append(df["request_rate"].iloc[0])
        selected_result["parallel_type"].append(df["parallel_type"].iloc[0])
        selected_result["model_ids"].append(df["model_id"].iloc[0])

        # 提取选定的列数据
        for column in selected_columns:
            if column in df.columns:
                if pd.api.types.is_numeric_dtype(df[column]):
                    if column == "avg_token_latency":
                        selected_result[column].append(df[df[column] != np.inf][column].mean())
                    else:
                        selected_result[column].append(df[column].min())
                else:
                    selected_result[column].append(df[column].iloc[0])
            else:
                selected_result[column].append(None)  # 如果列不存在，填充为 None

    return selected_result, selected_columns


def ttft_tpot_result_13b():
    data_types = ["fixed_result/share_gpt/llama2-13b/ttft_tpot/"]
    methods = [["fcfs", "l2r", "mlfq", "ours", "sjf"]]
    ttft_tpot_result_dir_names = generate_dir_names(base_dir, data_types, methods)
    ttft_tpot_result_dfs = get_result(ttft_tpot_result_dir_names)
    selected_result, selected_columns = extract_selected_results(ttft_tpot_result_dfs)

    long_df = process_selected_result(selected_result, selected_columns)
    csv_result_path = os.path.join(base_dir, data_types[0], "ttft_tpot_result.csv")
    long_df.to_csv(csv_result_path, index=False)

def ttft_tpot_result_70b():
    data_types = ["fixed_result/share_gpt/llama2-70b/ttft_tpot/"]
    methods = [["fcfs", "l2r", "mlfq", "ours", "sjf"]]
    ttft_tpot_result_dir_names = generate_dir_names(base_dir, data_types, methods)
    ttft_tpot_result_dfs = get_result(ttft_tpot_result_dir_names)
    selected_result, selected_columns = extract_selected_results(ttft_tpot_result_dfs)

    long_df = process_selected_result(selected_result, selected_columns)
    csv_result_path = os.path.join(base_dir, data_types[0], "ttft_tpot_result.csv")
    long_df.to_csv(csv_result_path, index=False)


ttft_tpot_result_13b()
ttft_tpot_result_70b()
