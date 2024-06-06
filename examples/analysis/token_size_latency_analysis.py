import argparse
import random
from typing import Tuple, Dict
from queue import Queue
from vllm import EngineArgs, LLMEngine, SamplingParams
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import pandas as pd
import multiprocessing as mp
from multiprocessing import Queue as MQueue
import os
from utils import Utils

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_requests() -> Dict[int, Tuple[str, SamplingParams, int]]:
    init_seq = {}
    prefill_seq = {}
    saved_seq = Utils.load_seq_from_file(
        BASE_DIR, "seq_data", "selected_seq.json"
    )
    for p_len in saved_seq:
        prompt_len = int(p_len)
        prompt = saved_seq[p_len]
        init_seq[prompt_len] = (
            prompt,
            SamplingParams(
                temperature=0.0,
                logprobs=1,
                min_tokens=160,
                max_tokens=161,
            ),
            prompt_len,
        )
        prefill_seq[prompt_len] = (
            prompt,
            SamplingParams(
                temperature=0.0,
                logprobs=1,
                min_tokens=32,
                max_tokens=32,
            ),
            prompt_len,
        )

    return init_seq, prefill_seq


def create_init_prompts(
    requests: Dict[
        str, Tuple[str, SamplingParams, int]
    ],  # prompt_len: Tuple[str, SamplingParams, int]
    prompts_queue: Queue,
    init_prompt_nums: int,
):
    selected_request = requests[
        1
    ]  # the prompt length of all initial sequences is 1
    requests = [selected_request] * init_prompt_nums
    for i in range(init_prompt_nums):
        prompts_queue.put(requests[i])


def add_new_request(
    prefill_requests: Dict[int, Tuple[str, SamplingParams, int]],
    prompts_queue: Queue,
    add_new_request_notice: Queue,
):
    add_new_request_notice.get()
    new_prompt = prefill_requests[1024]
    # print(new_prompt)
    for i in range(2):
        prompts_queue.put(new_prompt)


def initialize_engine(args: argparse.Namespace) -> LLMEngine:
    """Initialize the LLMEngine from the command line arguments."""
    engine_args = EngineArgs.from_cli_args(args)
    return LLMEngine.from_engine_args(engine_args)


def main(
    max_token_num: int,
    batch_size: int,
    result_queue: MQueue,
    enable_chunk_prefill: bool = False,
    policy: str = "fcfs",
):
    """Main function that sets up and runs the prompt processing."""
    parser = argparse.ArgumentParser(
        description="Demo on using the LLMEngine class directly"
    )
    parser = EngineArgs.add_cli_args(parser)
    args: argparse.Namespace = parser.parse_args()
    args.model = "meta-llama/Llama-2-13b-hf"
    args.max_num_seqs = batch_size
    args.scheduler_policy = policy
    # args.gpu_memory_utilization = 0.5
    if enable_chunk_prefill:
        args.enable_chunked_prefill = True
        args.max_num_batched_tokens = max_token_num
    try:
        init_requests, prefill_requests = get_requests()
        engine = initialize_engine(args)
    except Exception as e:
        print(e)
    for repeat_time in range(10):
        prompts_queue = Queue()
        add_new_request_notice = Queue()
        strategy = "hybrid"
        try:
            create_init_prompts(init_requests, prompts_queue, batch_size - 2)
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
                    enable_chunk_prefill=enable_chunk_prefill,
                    policy=policy,
                    repeat_time=repeat_time,
                    max_token_num=max_token_num,
                    random_seed=10,
                    prefill_mode="vertical",
                    insert_new_request=True,
                    insert_new_request_round=10,
                )
                executor.submit(
                    add_new_request,
                    prefill_requests,
                    prompts_queue,
                    add_new_request_notice,
                )
                # wait for all threads to finish
                executor.shutdown(wait=True)
        except Exception as e:
            print(e)


def skip_combination(df, batch_size, policy, random_seed, request_num):
    if df.shape[0] == 0:
        return False
    tmp = df[
        (df["batch_size"] == batch_size)
        & (df["policy"] == policy)
        & (df["random_seed"] == random_seed)
        & (df["request_num"] == request_num)
    ]
    if tmp.shape[0] == 0:
        return False
    return True


if __name__ == "__main__":
    test_type = "token_latency_swap_multi_tokens_limit_less_prefill_seq"
    # test_type = "token_latency_swap_pp"
    rerun = True 
    with mp.Manager() as manager:
        result_queue = manager.Queue()
        max_token_nums = [1024, 2048]
        batch_sizes = [8, 16, 32, 64, 128, 256, 512]
        total_iter_result, total_request_result = Utils.load_tmp_result(
            test_type, BASE_DIR
        )
        enable_chunk_prefill = True
        for batch_size in batch_sizes:
            for max_token_num in max_token_nums:
                for policy in ["fcfs"]:
                    for random_seed in range(10, 11, 1):
                        try:
                            if skip_combination(
                                total_iter_result,
                                batch_size,
                                policy,
                                random_seed,
                                max_token_num,
                            ):
                                print(
                                    f"skip {batch_size}, {policy},{random_seed}"
                                )
                                continue
                            with ProcessPoolExecutor(max_workers=2) as executor:
                                executor.submit(
                                    main,
                                    max_token_num,
                                    batch_size,
                                    result_queue,
                                    enable_chunk_prefill,
                                    policy,
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
