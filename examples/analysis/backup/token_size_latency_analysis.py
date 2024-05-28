import argparse
import time
from typing import List, Tuple, Optional, Dict
from queue import Queue
from vllm import EngineArgs, LLMEngine, RequestOutput, SamplingParams
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import pandas as pd
import multiprocessing as mp
from multiprocessing import Queue as MQueue
import random
import json
from transformers import PreTrainedTokenizerBase
import os
from vllm.transformers_utils.tokenizer import get_tokenizer
import uuid
from utils import Utils, RequestMetrics, IterMetrics

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_requests(
    dataset_path: str,
    tokenizer: PreTrainedTokenizerBase,
    fixed_output_len: Optional[int] = None,
    random_seed: Optional[int] = 42,
) -> List[Tuple[str, SamplingParams, int]]:
    if fixed_output_len is not None and fixed_output_len < 4:
        raise ValueError("output_len too small")

    init_seq = []
    prefill_seq = {}
    saved_seq = Utils.load_seq_from_file(BASE_DIR)
    init_seq.append(
            (
                saved_seq["init"][0][0],
                SamplingParams(
                    temperature=0.0,
                    logprobs=1,
                    # prompt_logprobs=1,
                    min_tokens=160,
                    max_tokens=161,
                    # stop_token_ids=[tokenizer.eos_token_id],
                ),
                saved_seq["init"][0][1],
            )
        )
    for i in range(len(saved_seq["prefill"])):
        prompt_len = saved_seq["prefill"][i][1]
        prompt = saved_seq["prefill"][i][0]
        prefill_seq[prompt_len] = (
            prompt,
            SamplingParams(
                temperature=0.0,
                logprobs=1,
                # prompt_logprobs=1,
                min_tokens=0,
                max_tokens=1,
                # stop_token_ids=[tokenizer.eos_token_id],
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
):
    """Add a new request to the queue, every 1 seconds."""

    prompt_request_length = list(requests.keys())
    print(prompt_request_length)
    for i in prompt_request_length:
        add_new_request_notice.get()
        new_prompt = requests[i]
        # print(new_prompt)
        prompts_queue.put(new_prompt)
    last_prompt = requests[16]
    last_prompt = (
        last_prompt[0],
        SamplingParams(
            temperature=0.0,
            logprobs=1,
            # prompt_logprobs=1,
            min_tokens=32,
            max_tokens=32,
            # stop_token_ids=[tokenizer.eos_token_id],
        ),
        last_prompt[2],
    )
    add_new_request_notice.get()
    prompts_queue.put(last_prompt)


def process_requests(
    engine: LLMEngine,
    prompts_queue: Queue,
    add_new_request_notice: Queue,
    strategy: str,
    result_queue: MQueue,
    batch_size: int,
    random_seed: int,
    enable_chunk_prefill: bool = False,
    policy: str = "fcfs",
    repeat_time: int = 1,
    max_token_num: int = 2048,
):
    """Continuously process a list of prompts and handle the outputs."""
    namespace = uuid.NAMESPACE_URL
    request_id = ""
    request_round = 0
    all_requests: List[int] = []
    prompt_lens = []
    request_metrics: Dict[str, RequestMetrics] = {}
    iter_metrics: List[IterMetrics] = []
    duplicate_prompt = 0
    try:
        while not prompts_queue.empty() or engine.has_unfinished_requests():
            while not prompts_queue.empty():
                prompt, sampling_params, prompt_len = prompts_queue.get()
                prompt_lens.append(prompt_len)
                if prompt_len in prompt_lens:
                    request_id = str(
                        uuid.uuid5(namespace, prompt + str(duplicate_prompt))
                    )
                    duplicate_prompt += 1
                else:
                    request_id = str(uuid.uuid5(namespace, prompt))
                engine.add_request(request_id, prompt, sampling_params)
                all_requests.append(request_id)
                st = time.time()
                request_metrics[request_id] = RequestMetrics(
                    request_id=request_id,
                    prompt_length=prompt_len,
                    decode_length=0,
                    request_start_time=st,
                    request_end_time=0,
                    batch_size=batch_size,
                    random_seed=random_seed,
                    enable_chunk_prefill=enable_chunk_prefill,
                    strategy=strategy,
                    policy=policy,
                )
            request_round = request_round + 1
            if request_round % 10 == 0:
                add_new_request_notice.put(1)
            st = time.time()
            try:
                request_outputs: List[RequestOutput] = engine.step()
            except Exception as e:
                print("error", e)
            et = time.time()
            request_metrics, iter_metric = Utils.parse_iter_metric(
                start_time=st,
                end_time=et,
                prompt_lens=prompt_lens,
                request_round=request_round,
                request_outputs=request_outputs,
                request_metrics=request_metrics,
                strategy=strategy,
                batch_size=batch_size,
                enable_chunk_prefill=enable_chunk_prefill,
                policy=policy,
                repeat_time=repeat_time,
                max_token_num=max_token_num,
            )
            iter_metrics.append(iter_metric)

    except Exception as e:
        print(e)
    try:
        request_result_metric = Utils.convert_request_metrics(request_metrics)
        iter_result_metric = Utils.convert_iter_metrics(iter_metrics)
    except Exception as e:
        print(e)
    result_queue.put((iter_result_metric, request_result_metric))
    return


def initialize_engine(args: argparse.Namespace) -> LLMEngine:
    """Initialize the LLMEngine from the command line arguments."""
    engine_args = EngineArgs.from_cli_args(args)
    return LLMEngine.from_engine_args(engine_args)


def main(
    max_token_num: int,
    batch_size: int,
    result_queue: MQueue,
    random_seed: int,
    enable_chunk_prefill: bool = False,
    policy: str = "fcfs",
):
    """Main function that sets up and runs the prompt processing."""
    parser = argparse.ArgumentParser(
        description="Demo on using the LLMEngine class directly"
    )
    dataset_path = (
        "/root/InferSchedule/dataset/ShareGPT_V3_unfiltered_cleaned_split.json"
    )
    parser = EngineArgs.add_cli_args(parser)
    args: argparse.Namespace = parser.parse_args()
    args.model = "meta-llama/Llama-2-13b-hf"
    tokenizer_id = args.tokenizer if args.tokenizer is not None else args.model
    tokenizer = get_tokenizer(
        tokenizer_id, trust_remote_code=args.trust_remote_code
    )
    args.max_num_seqs = batch_size
    args.scheduler_policy = policy
    # args.gpu_memory_utilization = 0.5
    if enable_chunk_prefill:
        args.enable_chunked_prefill = True
        args.max_num_batched_tokens = max_token_num
    try:
        init_requests, prefill_requests = get_requests(
            dataset_path=dataset_path,
            tokenizer=tokenizer,
            random_seed=random_seed,
        )
        engine = initialize_engine(args)
    except Exception as e:
        print(e)
    for repeat_time in range(10):
        prompts_queue = Queue()
        add_new_request_notice = Queue()
        strategy = "hybrid"
        try:
            create_init_prompts(init_requests, prompts_queue, batch_size - 1)
            print("start policy: ", policy)
            with ThreadPoolExecutor(max_workers=2) as executor:
                executor.submit(
                    process_requests,
                    engine,
                    prompts_queue,
                    add_new_request_notice,
                    strategy,
                    result_queue,
                    batch_size,
                    random_seed,
                    enable_chunk_prefill,
                    policy,
                    repeat_time,
                    max_token_num,
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
    os.environ["HF_TOKEN"] = "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"
    test_type = "token_latency_swap_multi_tokens_limit"
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
                                    random_seed,
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
