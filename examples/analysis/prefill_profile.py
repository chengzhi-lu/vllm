import argparse
import json
from typing import List, Tuple, Dict
from queue import Queue
import numpy as np
from vllm import AsyncEngineArgs, AsyncLLMEngine, EngineArgs, LLMEngine, SamplingParams
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import pandas as pd
import multiprocessing as mp
from multiprocessing import Queue as MQueue
import os
from utils import Utils
from rich import print, pretty
from parse_profile_result import ProfileParser
import re
pretty.install()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# print(BASE_DIR)


def get_requests() -> Dict[int, Tuple[str, SamplingParams, int]]:
    init_seq = {}
    saved_seq = Utils.load_seq_from_file(BASE_DIR,"seq_data", "selected_seq.json")
    for p_len in saved_seq:
        prompt_len = int(p_len)
        prompt = saved_seq[p_len]
        init_seq[prompt_len] = (
            prompt,
            SamplingParams(
                temperature=0.0,
                logprobs=1,
                min_tokens=1,
                max_tokens=1,
            ),
            prompt_len,
        )
    return init_seq


def create_init_prompts(
    seqs: Dict[int, Tuple[str, SamplingParams, int]],
    prompts_queue: Queue,
    init_prompt_nums: int,
    prefill_mode: str,
):
    if prefill_mode == "vertical":
        # create a batch whose size is $init_prompt_nums$ and each seq length is 1
        selected_seqs = [seqs[1]] * init_prompt_nums
    elif prefill_mode == "horizonal":
        # create a batch whose size is 1 and each seq length is init_prompt_nums
        selected_seqs = [seqs[init_prompt_nums]] *1
    for i in range(len(selected_seqs)):
        prompts_queue.put(selected_seqs[i])


def add_new_request(
    requests: List[Tuple[str, SamplingParams, int]],
    prompts_queue: Queue,
    add_new_request_notice: Queue,
    request_nums: int,
):
    """Add a new request to the queue, every 1 seconds."""

    add_new_request_notice.get()
    requests = requests * request_nums
    for i in range(request_nums):
        prompts_queue.put(requests[i])


def initialize_engine(args: argparse.Namespace) -> LLMEngine:
    """Initialize the LLMEngine from the command line arguments."""
    if  args.pipeline_parallel_size == 1:
        engine_args = EngineArgs.from_cli_args(args)
        engine = LLMEngine.from_engine_args(engine_args)
    else:
        args.engine_use_ray=False
        args.disable_log_requests=False
        args.max_log_len=None
        args.max_serving_time = 600
        engine_args = AsyncEngineArgs.from_cli_args(args)  
        engine = AsyncLLMEngine.from_engine_args(engine_args)
    return engine

def main(
    max_token_num: int,
    batch_size: int,
    result_queue: MQueue,
    model:str,
    enable_chunk_prefill: bool = False,
    policy: str = "fcfs",
    default_preemption_mode: str = "recompute",
    strategy: str = "full",
    prefill_mode: str = "vertical",
    pipeline_parallel_size: int = 1,
    tensor_parallel_size: int = 1,
):
    """Main function that sets up and runs the prompt processing."""
    parser = argparse.ArgumentParser(
        description="Demo on using the LLMEngine class directly"
    )

    parser = EngineArgs.add_cli_args(parser)
    args: argparse.Namespace = parser.parse_args()
    args.model = model
    if pipeline_parallel_size > 1:
        args.worker_use_ray=True
    args.pipeline_parallel_size=pipeline_parallel_size
    args.tensor_parallel_size = tensor_parallel_size
    args.max_num_seqs = batch_size
    args.scheduler_policy = policy
    args.default_preemption_mode = default_preemption_mode
    try:
        seqs = get_requests()
    except Exception as e:
        print(e)
    
    args.gpu_memory_utilization = 0.90
    if enable_chunk_prefill:
        args.enable_chunked_prefill = True
        args.max_num_batched_tokens = max_token_num
    try:
        engine = initialize_engine(args)
    except Exception as e:
        print(e)
        
    add_new_request_notice = Queue()
    print(f"start strategy: {strategy}, prefill_mode: {prefill_mode}")
    token_nums = [2**i for i in range(np.log2(max_token_num).astype(int))]
    for token_num in token_nums:
        for repeat_time in range(5):
            prompts_queue = Queue()
            if strategy == "hybrid":
                updated_token_num = int(token_num / 2)
                insert_new_request = True
            elif strategy == "full":
                updated_token_num = int(token_num)
                insert_new_request = False
            try:
                create_init_prompts(
                    seqs,
                    prompts_queue,
                    updated_token_num,
                    prefill_mode,
                )
                print(f"Sequence length: {updated_token_num}")
                with ThreadPoolExecutor(max_workers=2) as executor:
                    executor.submit(
                        Utils.process_requests,
                        engine=engine,
                        prompts_queue=prompts_queue,
                        add_new_request_notice=add_new_request_notice,
                        strategy=strategy,
                        result_queue=result_queue,
                        batch_size=updated_token_num,
                        enable_chunk_prefill=enable_chunk_prefill,
                        policy=policy,
                        repeat_time=repeat_time,
                        max_token_num=max_token_num,
                        random_seed=10,
                        prefill_mode=prefill_mode,
                        insert_new_request=insert_new_request,
                        insert_new_request_round=3,
                    )
                    # wait for all threads to finish
                    if insert_new_request:
                        executor.submit(
                            add_new_request,
                            seqs,
                            prompts_queue,
                            add_new_request_notice,
                            token_num - updated_token_num,
                        )
                    executor.shutdown(wait=True)
            except Exception as e:
                print(e)

def parse_file(file_path: str, profile_type: str):
        """
        解析日志文件并提取数据，保存为 CSV 文件。
        """
        # 从文件中读取文本
        with open(file_path, "r") as file:
            text = file.read()

        # 正则表达式匹配数字
        sequence_length_pattern = r"Sequence length: (\d+)"
        execution_time_pattern = r"Model execution time: (\d+\.\d+) ms"

        # 提取数据
        sequence_lengths = re.findall(sequence_length_pattern, text)
        execution_times = re.findall(execution_time_pattern, text)

        # 构建 DataFrame
        data = {
            "Sequence Length": sequence_lengths,
            "Model Execution Time": execution_times[1:],
        }
        df = pd.DataFrame(data)

        return df

def get_prefill_execution_time_model(df, model,pipeline_parallel_size):
    # use linearregression to fit the model
    df["Sequence Length"]=df["Sequence Length"].astype(float)
    df["Model Execution Time"]=df["Model Execution Time"].astype(float)
    df=df.groupby(['Sequence Length']).mean().reset_index()
    X = df[1:]["Sequence Length"].values
    y = df[1:]["Model Execution Time"].values
    coefficients = np.polyfit(X, y, 2)  # 2 表示二次函数
    a, b, c = coefficients
    
    prefill_profile_path = os.path.join(f"{BASE_DIR}",'prefill_profile.csv')
    if os.path.exists(prefill_profile_path):
        prefill_profile = pd.read_csv(prefill_profile_path)
    else:
        prefill_profile = pd.DataFrame(columns=["model_type", "parallel_type","num_instances", "a", "b", "c"])
    parallel_type = "pp" if pipeline_parallel_size > 1 else "single"

    tmp_df = pd.DataFrame(
        {
            "model_type": [model],
            "parallel_type": [parallel_type],
            "num_instances": [pipeline_parallel_size],
            "a": [a],
            "b": [b],
            "c": [c],
        }
    )


    prefill_profile = pd.concat([prefill_profile, tmp_df], ignore_index=True)
    prefill_profile.to_csv(prefill_profile_path, index=False)

    print(f"拟合结果: a = {a}, b = {b}, c = {c}")



if __name__ == "__main__":
    test_type = "profil_prefill_execution_time"
    log_file_path = "prefill_profile.log"
    parser = ProfileParser()
    # models = ["meta-llama/Llama-2-13b-chat-hf", "meta-llama/Llama-2-70b-chat-hf"]
    models=["meta-llama/Llama-2-13b-chat-hf"]
    # pipeline_parallel_sizes = [1,4]
    pipeline_parallel_sizes=[4]
    # tensor_parallel_sizes = [1,4]
    tensor_parallel_sizes = [1]
    with mp.Manager() as manager:
        result_queue = manager.Queue()
        max_token_nums = [8192]
        batch_sizes = [1]
        enable_chunk_prefill = True
        default_preemption_mode = "swap"
        default_policy = "fcfs"
        strategies = ["full"]
        # If prefill mode is horizonal, the sequences length is equals to the token nums, otherwise, the batch size equals to the token nums  # noqa: E501
        prefill_modes = ["horizonal"]
        for model in models:
            for pipeline_parallel_size in pipeline_parallel_sizes:
                for strategy in strategies:
                    for batch_size in batch_sizes:
                        for prefill_mode in prefill_modes:
                            for max_token_num in max_token_nums:
                                try:
                                    with ProcessPoolExecutor(max_workers=2) as executor:
                                        executor.submit(
                                            main,
                                            max_token_num=max_token_num,
                                            batch_size=batch_size,
                                            model=model,
                                            result_queue=result_queue,
                                            enable_chunk_prefill=enable_chunk_prefill,
                                            policy=default_policy,
                                            default_preemption_mode=default_preemption_mode,
                                            strategy=strategy,
                                            prefill_mode=prefill_mode,
                                            pipeline_parallel_size = pipeline_parallel_size,
                                        )
                                        executor.shutdown(wait=True)
                                except Exception as e:
                                    print(e)
                df = parse_file(log_file_path, test_type)
                get_prefill_execution_time_model(df, model, pipeline_parallel_size)