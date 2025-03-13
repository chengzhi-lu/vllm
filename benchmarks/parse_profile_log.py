import re
from datetime import datetime
import pandas as pd
import argparse
import numpy as np
import os
from scipy.optimize import curve_fit


def parse_profile_line(line):
    patterns = {
        "prefill_tokens": r"Prefill token nums: (\d+)",
        "decode_batch_size": r"Decode batch size: (\d+)",
        "decode_seq_lens": r"Decode Seq Lens: \[([\d, ]+)\]",
        "model_execution_time": r"Model execution time: ([\d\.]+)",
        "sampling_time": r"Sampling time: ([\d\.]+)",
    }
    result_dict = {}

    for key, pattern in patterns.items():
        match = re.search(pattern, line)
        if not match:
            print(f"Warning: Could not extract {key} from input text")
            return None

        value = match.group(1)

        if key == "decode_seq_lens":
            # Parse the array of sequence lengths
            result_dict[key] = [int(num.strip()) for num in value.split(",")]
        elif key in ["model_execution_time", "sampling_time"]:
            result_dict[key] = float(value)
        else:
            result_dict[key] = int(value)
    return result_dict


def prefill_model(df):
    if os.path.exists(os.path.join(dir_name, "prefill_profile.csv")):
        history_result = pd.read_csv(os.path.join(dir_name, "prefill_profile.csv"))
    else:
        history_result = pd.DataFrame(columns=["model_type", "parallel_type", "num_instances", "a", "b", "c"])
    prefill_df = df[df["decode_batch_size"] == 0]
    prefill_df = prefill_df[["prefill_tokens", "model_execution_time"]].groupby(["prefill_tokens"]).mean().reset_index()
    X = prefill_df[1:]["prefill_tokens"].values
    y = prefill_df[1:]["model_execution_time"].values
    coefficients = np.polyfit(X, y, 2)  # 2 表示二次函数
    a, b, c = coefficients
    new_profile_result = pd.DataFrame(
        {
            "model_type": [model_id],
            "parallel_type": [parallel_type],
            "num_instances": num_instances,
            "a": [a],
            "b": [b],
            "c": [c],
        }
    )
    history_result = pd.concat([history_result, new_profile_result])
    history_result.to_csv(os.path.join(dir_name, "prefill_profile.csv"), index=False)


def decode_model(df):
    if os.path.exists(os.path.join(dir_name, "decode_profile.csv")):
        history_result = pd.read_csv(os.path.join(dir_name, "decode_profile.csv"))
    else:
        history_result = pd.DataFrame(columns=["model_type", "parallel_type", "num_instances", "a", "b", "c"])
    decode_df = df[df["prefill_tokens"] == 0]
    decode_df = decode_df[decode_df['decode_seq_lens'].apply(len) > 0]
    def save_eval(x):
        try:
            return eval(x)
        except:
            return [0]
    decode_df["seq_len"] = decode_df["decode_seq_lens"].apply(lambda x: int(np.mean(save_eval(x))))
    decode_df = (
        decode_df[["decode_batch_size", "seq_len", "model_execution_time"]]
        .groupby(["decode_batch_size", "seq_len"])
        .mean()
        .reset_index()
    )
    X = decode_df[["decode_batch_size", "seq_len"]].values.reshape(-1)
    y = decode_df["model_execution_time"].values.reshape(-1)

    # X = [tuple(x) for x in X]
    # print(X)
    def func(x, a, b, c):
        bs = x[0]
        seq_len = x[1]
        return a * bs + b * bs * seq_len + c

    popt, pcov = curve_fit(func, X, y)
    new_profile_result = pd.DataFrame(
        {
            "model_type": [model_id],
            "parallel_type": [parallel_type],
            "num_instances": num_instances,
            "a": [popt[0]],
            "b": [popt[1]],
            "c": [popt[2]],
        }
    )
    history_result = pd.concat([history_result, new_profile_result])
    history_result.to_csv(os.path.join(dir_name, "decode_profile.csv"), index=False)


def sample_model(df):
    if os.path.exists(os.path.join(dir_name, "sample_profile.csv")):
        history_result = pd.read_csv(os.path.join(dir_name, "sample_profile.csv"))
    else:
        history_result = pd.DataFrame(columns=["model_type", "parallel_type", "num_instances", "a", "b"])

    def func(x, a, b):
        return a * x + b

    sample_df = df[df["prefill_tokens"] == 0]
    sample_df = sample_df[["decode_batch_size", "sampling_time"]].groupby(["decode_batch_size"]).mean().reset_index()
    X = sample_df["decode_batch_size"].values.reshape(-1)
    y = sample_df["sampling_time"].values.reshape(-1)
    popt, pcov = curve_fit(func, X, y)
    new_profile_result = pd.DataFrame(
        {
            "model_type": [model_id],
            "parallel_type": [parallel_type],
            "num_instances": num_instances,
            "a": [popt[0]],
            "b": [popt[1]],
        }
    )
    history_result = pd.concat([history_result, new_profile_result])
    history_result.to_csv(os.path.join(dir_name, "sample_profile.csv"), index=False)


def parse_log():
    selected_profile_lines = []
    with open(log_path, "r") as f:
        lines = f.readlines()
        selected_profile_lines = [parse_profile_line(line) for line in lines if "batch size" in line]
    profile_df = pd.DataFrame(selected_profile_lines)
    prefill_model(profile_df)
    decode_model(profile_df)
    sample_model(profile_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=str, default="fcfs")
    parser.add_argument("--swap-policy", type=str, default="full")
    parser.add_argument("--result-dir", type=str, default="/root/vllm/benchmarks/result")
    parser.add_argument(
        "--model",
        default="meta/Llama-2-13b",
        type=str,
        help="Name of the model.",
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Number of requests per second. If this is inf, "
        "then all the requests are sent at time 0. "
        "Otherwise, we use Poisson process to synthesize "
        "the request arrival times.",
    )
    parser.add_argument(
        "--execution-counter",
        type=int,
        default=0,
        help="Specify the execution counter.",
    )
    parser.add_argument(
        "--parallel-type",
        type=str,
        default="single",
    )
    parser.add_argument(
        "--num-instances",
        type=int,
        default=1,
    )
    args = parser.parse_args()
    base_dir = args.result_dir
    model_id = args.model
    policy = args.policy
    swap_policy = args.swap_policy
    execution_counter = args.execution_counter
    parallel_type = args.parallel_type
    num_instances = args.num_instances
    request_rate = args.request_rate
    days = "20250312"
    seconds = datetime.now().strftime("%H%M%S")
    base_model_id = model_id.split("/")[-1]
    dir_name = f"{base_dir}/{days}/{execution_counter}"
    file_name = f"{request_rate}qps-{base_model_id}-{seconds}-{policy}_profile.csv"
    result_path = f"{dir_name}/{file_name}"
    model=model_id.split("/")[1]
    log_path = f"result/profile/api_server_{policy}_{swap_policy}_{model}_{parallel_type}_profile.log"

    parse_log()
