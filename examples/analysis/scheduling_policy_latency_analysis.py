import argparse
from typing import List, Tuple, Dict
from queue import Queue
from vllm import EngineArgs, LLMEngine, SamplingParams
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import pandas as pd
import multiprocessing as mp
from multiprocessing import Queue as MQueue
import numpy as np
import os
from utils import Utils
from rich import print
from rich import pretty

pretty.install()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def save_seq_to_file(saved_seq):
    import json

    with open("saved_seq.json", "w") as f:
        json.dump(saved_seq, f)


def load_seq_from_file():
    import json

    if os.path.exists("/root/InferSchedule/examples/analysis/saved_seq.json"):
        with open(
            "/root/InferSchedule/examples/analysis/saved_seq.json", "r"
        ) as f:
            saved_seq = json.load(f)
        return saved_seq
    else:
        saved_seq = {"init": [], "prefill": []}
        return saved_seq


def get_requests() -> List[Tuple[str, SamplingParams, int]]:
    init_seq = {}
    saved_seq = Utils.load_seq_from_file(
        BASE_DIR, "seq_data", "selected_seq_new.json"
    )
    prefill_seq = {}
    for p_len in saved_seq:
        prompt_len = int(p_len)
        prompt = saved_seq[p_len]
        init_seq[prompt_len] = (
            prompt,
            SamplingParams(
                temperature=0.0,
                logprobs=1,
                min_tokens=80,
                max_tokens=81,
            ),
            prompt_len,
        )
        prefill_seq[prompt_len] = (
            prompt,
            SamplingParams(
                temperature=0.0,
                logprobs=1,
                min_tokens=0,
                max_tokens=1,
            ),
            prompt_len,
        )

    return init_seq, prefill_seq


def create_init_prompts(
    request: List[Tuple[str, SamplingParams, int]],
    prompts_queue: Queue,
    init_prompt_nums: int,
):
    request = request * init_prompt_nums
    for i in range(init_prompt_nums):
        prompts_queue.put(request[i])


def add_new_request(
    requests: Dict[int, Tuple[str, SamplingParams, int]],
    prompts_queue: Queue,
    add_new_request_notice: Queue,
    scheduling_policy: str,
    batch_size: int,
):
    """Add a new request to the queue, every 1 seconds."""

    request_orders = {
        64: {
            "fcfs": [
                requests[512],
                requests[512],
                requests[256],
                requests[256],
                requests[128],
            ],
            "stf": [
                requests[128],
                requests[256],
                requests[256],
                requests[512],
                requests[512],
            ],
            "lrf": [
                requests[128],
                requests[256],
                requests[512],
                requests[256],
                requests[512],
            ],
        },
        32: {
            "fcfs": [
                requests[512],
                requests[512],
                requests[256],
                requests[256],
                requests[128],
            ],
            "stf": [
                requests[128],
                requests[256],
                requests[256],
                requests[512],
                requests[512],
            ],
            "lrf": [
                requests[128],
                requests[256],
                requests[512],
                requests[256],
                requests[512],
            ],
        },
        128: {
            "fcfs": [
                requests[512],
                requests[512],
                requests[256],
                requests[256],
                requests[128],
            ],
            "stf": [
                requests[128],
                requests[256],
                requests[256],
                requests[512],
                requests[512],
            ],
            "lrf": [
                requests[256],
                requests[256],
                requests[512],
                requests[128],
                requests[512],
            ],
        },
    }
    if scheduling_policy not in request_orders[batch_size]:
        raise ValueError(
            f"Scheduling policy {scheduling_policy} is not supported."
        )
    add_new_request_notice.get()
    for new_prompt in request_orders[batch_size][scheduling_policy]:
        prompts_queue.put(new_prompt)


def calc_cv(prompt_lens: List[int]):
    if len(prompt_lens) == 0:
        return 0
    prompt_lens_mean = np.mean(prompt_lens)
    prompt_lens_std = np.std(prompt_lens)
    return prompt_lens_std / prompt_lens_mean


def initialize_engine(args: argparse.Namespace) -> LLMEngine:
    """Initialize the LLMEngine from the command line arguments."""
    engine_args = EngineArgs.from_cli_args(args)
    return LLMEngine.from_engine_args(engine_args)


def main(
    max_token_num: int,
    batch_size: int,
    result_queue: MQueue,
    random_seed: int = 10,
    enable_chunk_prefill: bool = False,
    policy: str = "fcfs",
    default_preemption_mode: str = "recompute",
    pipeline_parallel_size: int = 1,
):
    """Main function that sets up and runs the prompt processing."""
    parser = argparse.ArgumentParser(
        description="Demo on using the LLMEngine class directly"
    )
    parser = EngineArgs.add_cli_args(parser)
    args: argparse.Namespace = parser.parse_args()
    args.model = "meta-llama/Llama-2-13b-hf"
    args.max_num_seqs = batch_size
    args.default_preemption_mode = default_preemption_mode
    args.pipeline_parallel_size = pipeline_parallel_size
    # args.gpu_memory_utilization = 0.5
    if enable_chunk_prefill:
        args.enable_chunked_prefill = True
        args.max_num_batched_tokens = max_token_num
    try:
        init_requests, prefill_requests = get_requests()
        engine = initialize_engine(args)
    except Exception as e:
        print(e)
    for repeat_time in range(20):
        prompts_queue = Queue()
        add_new_request_notice = Queue()
        strategy = "hybrid"
        try:
            create_init_prompts(init_requests, prompts_queue, batch_size - 3)
            print("start policy: ", policy)
            with ThreadPoolExecutor(max_workers=2) as executor:
                executor.submit(
                    Utils.process_requests,
                    engine=engine,
                    prompts_queue=prompts_queue,
                    add_new_request_notice=add_new_request_notice,
                    strategy=strategy,
                    result_queue=result_queue,
                    batch_size=batch_size,
                    random_seed=random_seed,
                    enable_chunk_prefill=enable_chunk_prefill,
                    policy=policy,
                    repeat_time=repeat_time,
                    max_token_num=max_token_num,
                    prefill_mode="vertical",
                    insert_new_requests=True,
                    insert_new_request_round=5,
                )
                executor.submit(
                    add_new_request,
                    prefill_requests,
                    prompts_queue,
                    add_new_request_notice,
                    policy,
                    batch_size,
                )
                # wait for all threads to finish
                executor.shutdown(wait=True)
        except Exception as e:
            print(e)


def skip_combination(df, batch_size, policy="fcfs", random_seed=10):
    if df.shape[0] == 0:
        return False
    tmp = df[
        (df["batch_size"] == batch_size)
        & (df["policy"] == policy)
        & (df["random_seed"] == random_seed)
    ]
    if tmp.shape[0] == 0:
        return False
    return True


def get_test_params(test_type):
    if test_type == "scheduling_policy_swap":
        return {
            "batch_sizes": [32, 64, 128, 256],
            "policies": ["fcfs", "stf", "lrf"],
            "pipeline_parallel_size": 1,
            "default_preemption_mode": "swap",
        }
    elif test_type == "scheduling_policy_swap_pp":
        return {
            "batch_sizes": [32, 64, 128, 256],
            "policies": ["fcfs", "stf", "lrf"],
            "pipeline_parallel_size": 2,
            "default_preemption_mode": "swap",
        }
    elif test_type == "scheduling_policy":
        return {
            "batch_sizes": [32, 64, 128, 256],
            "policies": ["fcfs", "stf", "lrf"],
            "pipeline_parallel_size": 2,
            "default_preemption_mode": "recompute",
        }
    else:
        raise ValueError(f"test_type {test_type} is not supported")


if __name__ == "__main__":
    test_type = "scheduling_policy_swap"
    rerun = True
    # test_type = "scheduling_policy_swap_pp"
    with mp.Manager() as manager:
        result_queue = manager.Queue()
        params = get_test_params(test_type)
        batch_sizes = params["batch_sizes"]
        policies = params["policies"]
        max_token_nums = [1024]
        default_preemption_mode = params["default_preemption_mode"]
        pipeline_parallel_size = params["pipeline_parallel_size"]
        total_iter_result, total_request_result = Utils.load_tmp_result(
            test_type, BASE_DIR
        )
        enable_chunk_prefill = True

        for batch_size in batch_sizes:
            for max_token_num in max_token_nums:
                for policy in policies:
                    for random_seed in range(10, 11, 1):
                        try:
                            if (
                                skip_combination(
                                    total_iter_result,
                                    batch_size,
                                    policy,
                                    random_seed,
                                )
                                and not rerun
                            ):
                                print(
                                    f"skip {batch_size}, {policy},{random_seed}"
                                )
                                continue
                            with ProcessPoolExecutor(max_workers=2) as executor:
                                executor.submit(
                                    main,
                                    max_token_num=max_token_num,
                                    batch_size=batch_size,
                                    result_queue=result_queue,
                                    enable_chunk_prefill=enable_chunk_prefill,
                                    policy=policy,
                                    default_preemption_mode=default_preemption_mode,
                                    pipeline_parallel_size=pipeline_parallel_size,
                                )
                                executor.shutdown(wait=True)
                            while not result_queue.empty():
                                item = result_queue.get()
                                iter_result, request_result = item[0], item[1]
                                total_iter_result = pd.concat(
                                    [total_iter_result, iter_result]
                                )
                                total_request_result = pd.concat(
                                    [total_request_result, request_result]
                                )
                            if len(total_iter_result) > 0:
                                Utils.save_tmp_result(
                                    total_iter_result,
                                    total_request_result,
                                    test_type,
                                    BASE_DIR,
                                )
                        except Exception as e:
                            print(e)
        if len(total_iter_result) > 0:
            Utils.save_result(
                total_iter_result,
                total_request_result,
                enable_chunk_prefill,
                test_type,
                rerun,
                BASE_DIR,
            )
