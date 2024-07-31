import re
from datetime import datetime
import pandas as pd
import argparse


def parse_line(line):
    input_str = (line.split("]")[1].replace("reqs",
                                            "").replace("tokens/s",
                                                        "").replace("%", ""))
    pattern = re.compile(r"(\w[\w\s]*\w):\s*([^,]+)")
    matches = pattern.findall(input_str)
    result_dict = {key.strip(): value.strip() for key, value in matches}
    return result_dict


def parse_detailed_line(line):
    input_str = line.split("]")[1].replace(" s,", ",")

    pattern = re.compile(r"(\w[\w\s]*\w):\s*([^,]+)")
    matches = pattern.findall(input_str)
    result_dict = {key.strip(): value.strip() for key, value in matches}
    return result_dict


def parse_log():
    selected_lines = []
    with open(log_path, "r") as f:
        lines = f.readlines()
        selected_lines = [
            parse_line(line) for line in lines if "generation" in line
        ]
        selected_detaile_lines = [
            parse_detailed_line(line) for line in lines
            if "iteration number" in line
        ]
    df = pd.DataFrame(selected_lines)
    detailed_df = pd.DataFrame(selected_detaile_lines)
    df.to_csv(result_path, index=False)
    detailed_df.to_csv(result_path.replace(".csv", "_detailed.csv"),
                       index=False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=str, default="fcfs", required=True)
    parser.add_argument("--swap-policy",
                        type=str,
                        default="full",
                        required=True)
    parser.add_argument("--result-dir",
                        type=str,
                        default="/root/vllm/benchmarks/result")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
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
        "--swap-out-partial-rate",
        type=float,
        default=1.0,
        help="The rate at which we swap out partial models.",
    )
    args = parser.parse_args()
    base_dir = args.result_dir
    model_id = args.model
    policy = args.policy
    swap_policy = args.swap_policy
    execution_counter = args.execution_counter
    request_rate = args.request_rate
    days = datetime.now().strftime("%Y%m%d")
    seconds = datetime.now().strftime("%H%M%S")
    base_model_id = model_id.split("/")[-1]
    dir_name = f"{base_dir}/{days}/{execution_counter}"
    file_name = f"{request_rate}qps-{base_model_id}-{seconds}-{policy}.csv"
    result_path = f"{dir_name}/{file_name}"
    log_file_path = f"api_server_{policy}_{swap_policy}.log"
    return log_file_path, result_path


if __name__ == "__main__":
    log_path, result_path = parse_args()
    parse_log()
