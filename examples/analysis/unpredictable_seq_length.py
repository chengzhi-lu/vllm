import argparse
from typing import Tuple, Dict
import time
import random
from queue import Queue

from vllm import EngineArgs, LLMEngine, SamplingParams
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import pandas as pd
import multiprocessing as mp
from multiprocessing import Queue as MQueue
import os
from utils import Utils
from vllm.transformers_utils.tokenizer import get_tokenizer
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_all_data():
    dataset_path = "/root/vllm/dataset/ShareGPT_V3_unfiltered_cleaned_split.json"
    parser = argparse.ArgumentParser(
        description="Demo on using the LLMEngine class directly"
    )
    parser = EngineArgs.add_cli_args(parser)
    args: argparse.Namespace = parser.parse_args()
    args.model = "meta-llama/Llama-2-13b-hf"
    tokenizer_id = args.tokenizer if args.tokenizer is not None else args.model
    tokenizer = get_tokenizer(
        tokenizer_id, trust_remote_code=args.trust_remote_code
    )
    with open(dataset_path) as f:
        dataset = json.load(f)
        # Filter out the conversations with less than 2 turns.
        # Only keep the first two turns of each conversation.
        # Shuffle the dataset.
        random.seed(10)
        random.shuffle(dataset)
        dataset = [data["conversations"][0]["value"] for data in dataset if len(data["conversations"]) >= 2 ]
        for data in dataset:
            prompt_length = len(tokenizer(data).input_ids)
            if prompt_length < 4080:
                yield (data,len(tokenizer(data).input_ids))


def put_requests(prompt_queue: Queue):
    count = 0
    total_count = 0
    for prompt, seq_len in get_all_data():
        prompt_queue.put(
            (
                prompt,
                SamplingParams(
                    temperature=0.01,
                    logprobs=1,
                    min_tokens=1,
                    max_tokens=10000,
                ),
                seq_len
            )
        )
        count += 1
        total_count += 1
        if count == 32:
            time.sleep(2)
            count = 0
        if total_count  == 16000:
            break
        




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
        engine = initialize_engine(args)
    except Exception as e:
        print(e)
    prompts_queue = Queue()
    add_new_request_notice = Queue()
    strategy = "full"
    try:
        print("start policy: ", policy)
        with ThreadPoolExecutor(max_workers=2) as executor:
            executor.submit(
                put_requests,
                prompts_queue,
            )
            while prompts_queue.qsize()==0:
                time.sleep(1)
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
                repeat_time=1,
                max_token_num=max_token_num,
                random_seed=10,
                prefill_mode="vertical",
                insert_new_request=False,
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
    test_type = "new_unpredict_seq_length"
    # test_type = "token_latency_swap_pp"
    rerun = True 
    with mp.Manager() as manager:
        result_queue = manager.Queue()
        max_token_num = 2048
        batch_size = 64 
        policy = "fcfs"
        random_seed = 10
        total_iter_result, total_request_result = Utils.load_tmp_result(
            test_type, BASE_DIR
        )
        enable_chunk_prefill = True
        # for prompt_len in prompt_length:
        try:
            if skip_combination(
                total_iter_result,
                batch_size,
                policy,
                random_seed,
                max_token_num,
            ):
                print(f"skip {batch_size}, {policy},{random_seed}")
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
                _, request_result = item[0], item[1]
                # total_iter_result = pd.concat([total_iter_result, iter_result])
                total_request_result = pd.concat(
                    [total_request_result, request_result]
                )
                total_request_result.to_csv("tmp_result.csv")
            if len(total_request_result) > 0:
                Utils.save_tmp_result(
                    total_iter_result,
                    total_request_result,
                    test_type,
                    BASE_DIR,
                )
        except Exception as e:
            print(e)
        if len(total_request_result) > 0:
            Utils.save_result(
                total_iter_result,
                total_request_result,
                enable_chunk_prefill,
                test_type,
                rerun,
                BASE_DIR,
            )
