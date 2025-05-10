import pandas as pd

def add_num_annotation(ax, rotation=0, fontsize=10):
    for _p in ax.patches:
        if _p.get_height() == 0:
            continue
        ax.annotate(
            str(round(_p.get_height(), 2)),
            (_p.get_x() + _p.get_width() / 2.0, _p.get_height() * 1.05),
            ha="center",
            va="center",
            xytext=(0, 6),
            textcoords="offset points",
            rotation=rotation,
            fontsize=fontsize,
        )
def get_metric_ratio(df,metric_min_result):
    min_result = metric_min_result[metric_min_result['Metric'] == df['Metric'].values[0]]['Value'].values[0]
    df["Ratio"] = round(df["Value"] / min_result,2)
    return df
def get_metric_ratio_result(df, columns):
    df[columns] = round(df[columns] /1 ,2)
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
    elif  detailed and "_detailed" in file_path:
        df = pd.read_csv(file_path)
    else:
        return None, None

    for policy in schedule_policies:
        if policy in file_path:
            return policy.upper(), df
    return None, None


def load_execute_results(dir_names, schedule_policies,qps, detailed):
    """加载执行结果并返回字典"""
    execute_result_dfs = {}
    for dir_name in dir_names:
        for file_name in os.listdir(dir_name):
            file_path = os.path.join(dir_name, file_name)
            policy, df = process_file(file_path, schedule_policies,qps,detailed)
            if policy and df is not None:
                execute_result_dfs[policy] = df
    return execute_result_dfs



def get_result(e2e_result_dir_names):
    json_files = []
    for dir_name in e2e_result_dir_names:
        json_files.extend(glob(os.path.join(dir_name, "*.json")))
    e2e_result_dfs = {}
    for file_path in json_files:
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
