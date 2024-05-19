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
import numpy as np
from transformers import PreTrainedTokenizerBase
import os
from vllm.transformers_utils.tokenizer import get_tokenizer
from dataclasses import dataclass
import uuid


@dataclass
class RequestMetrics:
    request_id: str
    prompt_length: int
    decode_length: int
    request_start_time: float
    request_end_time: float
    batch_size: int
    request_num: int
    random_seed: int
    strategy: str
    enable_chunk_prefill: bool
    policy: str


@dataclass
class IterMetrics:
    request_round: str
    latency: float
    request_num_each_round: int
    num_tokens: int
    token_chunk_size: int
    num_running_to_waiting: int
    num_waiting_to_running: int
    recomputed_token_nums: int
    prefill_token_num_each_request: str
    cvs: float
    current_stage: str
    wasted_block_sizes: int
    total_block_sizes: int
    throughput: float
    batch_size: int
    request_num: int
    random_seed: int
    strategy: str
    enable_chunk_prefill: bool
    policy: str
    repeat_time: int


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


def get_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    fixed_output_len: Optional[int] = None,
    random_seed: Optional[int] = 42,
    enable_chunk_prefill: bool = True,
) -> List[Tuple[str, SamplingParams, int]]:
    if fixed_output_len is not None and fixed_output_len < 4:
        raise ValueError("output_len too small")

    init_seq = []
    prefill_seq = {}
    saved_seq = load_seq_from_file()

    is_power_of_two = lambda x: x > 0 and (x & (x - 1)) == 0
    if len(saved_seq["init"]) == 0:
        # Load the dataset.
        with open(dataset_path) as f:
            dataset = json.load(f)
        # Filter out the conversations with less than 2 turns.
        dataset = [data for data in dataset if len(data["conversations"]) >= 2]
        # Only keep the first two turns of each conversation.
        dataset = [
            (
                data["conversations"][0]["value"],
                data["conversations"][1]["value"],
            )
            for data in dataset
        ]

        # Shuffle the dataset.
        random.seed(random_seed)
        random.shuffle(dataset)
        for i in range(len(dataset)):
            if (
                len(prefill_seq) == 9
                and len(init_seq) == 1
                and max(prefill_seq.keys()) == 2048
                and min(prefill_seq.keys()) == 4
            ):
                break

            # Tokenize the prompts and completions.
            prompt = dataset[i][0]
            prompt_token_ids = tokenizer(prompt).input_ids
            completion = dataset[i][1]
            completion_token_ids = tokenizer(completion).input_ids
            prompt_len = len(prompt_token_ids)
            output_len = (
                len(completion_token_ids)
                if fixed_output_len is None
                else fixed_output_len
            )

            if len(init_seq) == 0 and prompt_len < 8 and output_len <= 1024:
                init_seq.append(
                    (
                        prompt,
                        SamplingParams(
                            temperature=0.0,
                            logprobs=1,
                            # prompt_logprobs=1,
                            min_tokens=300,
                            max_tokens=301,
                            # stop_token_ids=[tokenizer.eos_token_id],
                        ),
                        prompt_len,
                    )
                )
                saved_seq["init"].append(
                    (
                        prompt,
                        prompt_len,
                    )
                )
            else:
                if is_power_of_two(prompt_len):
                    prefill_seq[prompt_len] = (
                        prompt,
                        SamplingParams(
                            temperature=0.0,
                            logprobs=1,
                            # prompt_logprobs=1,
                            min_tokens=6,
                            max_tokens=7,
                            # stop_token_ids=[tokenizer.eos_token_id],
                        ),
                        prompt_len,
                    )
                    saved_seq["prefill"].append(
                        (
                            prompt,
                            prompt_len,
                        )
                    )
        save_seq_to_file(saved_seq)
    else:
        init_seq.append(
            (
                saved_seq["init"][0][0],
                SamplingParams(
                    temperature=0.0,
                    logprobs=1,
                    # prompt_logprobs=1,
                    min_tokens=80,
                    max_tokens=81,
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
    scheduling_policy: str,
    batch_size: int,
):
    """Add a new request to the queue, every 1 seconds."""

    prompt_request_length = list(requests.keys())
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


def parse_iter_metric(
    start_time: int,
    end_time: int,
    prompt_lens: List[int],
    request_round: int,
    request_outputs: List[RequestOutput],
    request_metrics: Dict[str, RequestMetrics],
    strategy: str,
    policy: str,
    batch_size: int,
    enable_chunk_prefill: bool,
    repeat_time: int,
):
    # parse the prefill token numbers and decode token numbers
    latency = end_time - start_time
    prefill_token_nums = 0
    decode_token_nums = 0
    prefill_stage = False
    decode_stage = False
    prefill_token_num_each_request = []
    wasted_block_sizes = 0
    total_block_sizes = 0
    num_running_to_waiting = 0
    num_waiting_to_running = 0
    recomputed_token_nums = 0
    request_num_each_round = 0
    for request_output in request_outputs:
        request_num_each_round += 1
        request_id = request_output.request_id
        num_waiting_to_running = request_output.num_running_to_waiting
        num_running_to_waiting = request_output.num_waiting_to_running
        recomputed_token_nums = request_output.recomputed_token_nums
        if len(request_output.outputs[0].token_ids) == 0:
            # prefill stage
            prefill_token_nums += request_output.token_chunk_size
            prefill_token_num_each_request.append(
                (request_id, request_output.token_chunk_size)
            )
            prefill_stage = True
        elif len(request_output.outputs[0].token_ids) == 1:
            prefill_token_nums += request_output.token_chunk_size
            prefill_token_num_each_request.append(
                (request_id, request_output.token_chunk_size)
            )
            decode_token_nums += 1
            prefill_stage = True
        else:
            decode_token_nums += 1
            decode_stage = True
        if request_output.finished:
            request_metrics[f"{request_id}"].request_end_time = time.time()
            request_metrics[f"{request_id}"].decode_length = len(
                request_output.outputs[0].token_ids
            )
        wasted_block_sizes += request_output.wasted_block_size
        total_block_sizes += request_output.total_block_size
    current_stage = "hybrid"
    if prefill_stage and decode_stage:
        current_stage = "hybrid"
    elif prefill_stage and not decode_stage:
        current_stage = "prefill"
    else:
        current_stage = "decode"

    iter_metric = IterMetrics(
        request_round=request_round,
        latency=latency,
        throughput=(prefill_token_nums + decode_token_nums) / latency,
        request_num_each_round=request_num_each_round,
        prefill_token_num_each_request=prefill_token_num_each_request,
        num_tokens=prefill_token_nums + decode_token_nums,
        token_chunk_size=request_outputs[0].token_chunk_size,
        cvs=calc_cv(prompt_lens),
        batch_size=batch_size,
        request_num=len(request_metrics),
        random_seed=request_round,
        strategy=strategy,
        enable_chunk_prefill=enable_chunk_prefill,
        policy=policy,
        num_running_to_waiting=num_running_to_waiting,
        num_waiting_to_running=num_waiting_to_running,
        recomputed_token_nums=recomputed_token_nums,
        current_stage=current_stage,
        wasted_block_sizes=wasted_block_sizes,
        total_block_sizes=total_block_sizes,
        repeat_time=repeat_time,
    )

    return request_metrics, iter_metric


def convert_request_metrics(request_metrics: Dict[str, RequestMetrics]):
    request_metrics = request_metrics.values()
    request_metrics = list(request_metrics)
    df = pd.DataFrame([vars(m) for m in request_metrics])
    df.set_axis(RequestMetrics.__dataclass_fields__, axis=1)
    return df


def convert_iter_metrics(iter_metrics: List[IterMetrics]):
    df = pd.DataFrame([vars(m) for m in iter_metrics])
    df.set_axis(IterMetrics.__dataclass_fields__, axis=1)
    return df


def process_requests(
    engine: LLMEngine,
    prompts_queue: Queue,
    add_new_request_notice: Queue,
    strategy: str,
    result_queue: MQueue,
    batch_size: int,
    request_num: int,
    random_seed: int,
    enable_chunk_prefill: bool = False,
    policy: str = "fcfs",
    repeat_time: int = 1,
):
    """Continuously process a list of prompts and handle the outputs."""
    namespace = uuid.NAMESPACE_URL
    request_id = ""
    request_round = 0
    all_requests: List[int] = []
    prompt_lens = []
    request_metrics: Dict[str, RequestMetrics] = {}
    iter_metrics: List[IterMetrics] = []
    total_start_time = time.time()
    duplicate_prompt = 0
    try:
        while not prompts_queue.empty() or engine.has_unfinished_requests():
            while not prompts_queue.empty():
                prompt, sampling_params, prompt_len = prompts_queue.get()
                if prompt_len in prompt_lens:
                    request_id = str(
                        uuid.uuid5(namespace, prompt + str(duplicate_prompt))
                    )
                    duplicate_prompt += 1
                else:
                    request_id = str(uuid.uuid5(namespace, prompt))
                prompt_lens.append(prompt_len)

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
                    request_num=request_num,
                    random_seed=random_seed,
                    enable_chunk_prefill=enable_chunk_prefill,
                    strategy=strategy,
                    policy=policy,
                )
            request_round = request_round + 1
            if request_round % 40 == 0:
                add_new_request_notice.put(1)
            st = time.time()
            try:
                request_outputs: List[RequestOutput] = engine.step()
            except Exception as e:
                print("error", e)
            et = time.time()
            request_metrics, iter_metric = parse_iter_metric(
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
            )
            iter_metrics.append(iter_metric)

    except Exception as e:
        print(e)
    try:
        request_result_metric = convert_request_metrics(request_metrics)
        iter_result_metric = convert_iter_metrics(iter_metrics)
    except Exception as e:
        print(e)
    result_queue.put((iter_result_metric, request_result_metric))
    return


def initialize_engine(args: argparse.Namespace) -> LLMEngine:
    """Initialize the LLMEngine from the command line arguments."""
    engine_args = EngineArgs.from_cli_args(args)
    return LLMEngine.from_engine_args(engine_args)


def main(
    request_num: int,
    batch_size: int,
    result_queue: MQueue,
    random_seed: int,
    enable_chunk_prefill: bool = False,
    policy: str = "fcfs",
    default_preemption_mode: str = "recompute",
    pipeline_parallel_size: int = 1,
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
    args.default_preemption_mode = default_preemption_mode
    args.pipeline_parallel_size = pipeline_parallel_size
    # args.gpu_memory_utilization = 0.5
    if enable_chunk_prefill:
        args.enable_chunked_prefill = True
        args.max_num_batched_tokens = 1024
    try:
        init_requests, prefill_requests = get_requests(
            dataset_path=dataset_path,
            num_requests=request_num,
            tokenizer=tokenizer,
            random_seed=random_seed,
            enable_chunk_prefill=enable_chunk_prefill,
        )
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
                    process_requests,
                    engine,
                    prompts_queue,
                    add_new_request_notice,
                    strategy,
                    result_queue,
                    batch_size,
                    request_num,
                    random_seed,
                    enable_chunk_prefill,
                    policy,
                    repeat_time,
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


def save_result(
    total_iter_result, total_request_result, enable_chunk_prefill, test_type
):
    chunks = ""
    if not enable_chunk_prefill:
        chunks = "not_chunks_"

    request_result_tmp_path = f"data/tmp/{chunks}request_result_{test_type}.csv"
    request_result_path = (
        f"data/request_level/{chunks}new_request_result_{test_type}_fcfs.csv"
    )
    iter_result_tmp_path = (
        f"data/tmp/{chunks}new_iter_result_{test_type}_policy.csv"
    )
    iter_result_path = (
        f"data/iter_level/{chunks}new_iter_result_{test_type}_policy.csv"
    )

    total_request_result.to_csv(request_result_tmp_path, index=False)
    total_iter_result.to_csv(iter_result_tmp_path, index=False)
    if os.path.exists(iter_result_path):
        tmp = pd.read_csv(iter_result_path)
        total_iter_result_policy = total_request_result["policy"].unique()
        total_iter_result_batch_size = total_request_result[
            "batch_size"
        ].unique()
        # remove all rows that have the same policy and batch_size
        tmp = tmp[
            ~(tmp["policy"].isin(total_iter_result_policy))
            | ~(tmp["batch_size"].isin(total_iter_result_batch_size))
        ]
        total_iter_result = pd.concat([tmp, total_iter_result])
    if os.path.exists(request_result_path):
        tmp = pd.read_csv(request_result_path)
        total_request_result_policy = total_request_result["policy"].unique()
        total_request_result_batch_size = total_request_result[
            "batch_size"
        ].unique()
        tmp = tmp[
            ~(tmp["policy"].isin(total_request_result_policy))
            | ~(tmp["batch_size"].isin(total_request_result_batch_size))
        ]
        total_request_result = pd.concat([tmp, total_request_result])
    total_request_result.to_csv(
        request_result_path,
        index=False,
    )
    total_iter_result.to_csv(iter_result_path, index=False)
    os.remove(request_result_tmp_path)
    os.remove(iter_result_tmp_path)


def load_tmp_result(test_type):
    iter_result_path = f"data/tmp/tmp_{test_type}_iter_level.csv"
    request_result_path = f"data/tmp/tmp_{test_type}_request_level.csv"
    if os.path.exists(iter_result_path) and os.path.exists(request_result_path):
        total_iter_result = pd.read_csv(iter_result_path)
        total_request_result = pd.read_csv(request_result_path)
    else:
        total_iter_result = pd.DataFrame()
        total_request_result = pd.DataFrame()
    return total_iter_result, total_request_result


def skip_combination(df, batch_size, policy, random_seed):
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
    os.environ["HF_TOKEN"] = "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"
    # test_type = "scheduling_policy_swap"
    test_type = "scheduling_policy_swap_pp"
    with mp.Manager() as manager:
        result_queue = manager.Queue()
        params = get_test_params(test_type)
        request_nums = [2048]
        batch_sizes = params["batch_sizes"]
        policies = params["policies"]
        default_preemption_mode = params["default_preemption_mode"]
        pipeline_parallel_size = params["pipeline_parallel_size"]
        total_iter_result, total_request_result = load_tmp_result(test_type)
        enable_chunk_prefill = True

        for batch_size in batch_sizes:
            for request_num in request_nums:
                for policy in policies:
                    for random_seed in range(10, 11, 1):
                        try:
                            if skip_combination(
                                total_iter_result,
                                batch_size,
                                policy,
                                random_seed,
                            ):
                                print(
                                    f"skip {batch_size}, {policy},{random_seed}"
                                )
                                continue
                            with ProcessPoolExecutor(max_workers=2) as executor:
                                executor.submit(
                                    main,
                                    request_num,
                                    batch_size,
                                    result_queue,
                                    random_seed,
                                    enable_chunk_prefill,
                                    policy,
                                    default_preemption_mode,
                                    pipeline_parallel_size,
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
                                total_iter_result.to_csv(
                                    f"data/tmp/tmp_{test_type}_iter_level.csv",
                                    index=False,
                                )
                                total_request_result.to_csv(
                                    f"data/tmp/tmp_{test_type}_request_level.csv",
                                    index=False,
                                )
                        except Exception as e:
                            print(e)
        if len(total_iter_result) > 0:
            save_result(
                total_iter_result,
                total_request_result,
                enable_chunk_prefill,
                test_type,
            )
