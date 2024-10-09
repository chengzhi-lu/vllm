import argparse
from queue import Queue

from vllm import EngineArgs, LLMEngine, SamplingParams
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import pandas as pd
import multiprocessing as mp
from multiprocessing import Event
from multiprocessing import Queue as MQueue
import os
from utils import Utils
from rich import print
from rich import pretty
import traceback
import time
import hashlib
pretty.install()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))



def get_requests():
    saved_seq = Utils.load_seq_from_file(
        BASE_DIR, "seq_data", "selected_seq_full.json"
    )
    for prompt_len, prompt in saved_seq.items():
        yield (prompt, prompt_len)


def put_requests(stop_event: Event, prompt_queue: Queue, result_queue: Queue):
    prompt_len = []
    exec_time = []
    total_count = 10000
    count = 0
    for prompt, seq_len in get_requests():
        prompt_queue.put(
            (
                prompt,
                SamplingParams(
                    temperature=0.01,
                    logprobs=1,
                    min_tokens=1,
                    max_tokens=1,
                ),
                seq_len
            )
        )
        st = time.time()
        et = result_queue.get()
        prompt_len.append(seq_len) 
        exec_time.append(et - st)   
        count += 1
        if count == total_count:
            break
        time.sleep(0.1)
    stop_event.set()
    result =pd.DataFrame({"prompt_len": prompt_len, "exec_time": exec_time})
    result.to_csv(os.path.join(BASE_DIR, "exec_time.csv"), index=False)
    return 


def initialize_engine(args: argparse.Namespace) -> LLMEngine:
    """Initialize the LLMEngine from the command line arguments."""
    engine_args = EngineArgs.from_cli_args(args)
    return LLMEngine.from_engine_args(engine_args)



def test_prefilling_time(engine, stop_event: mp.Event, result_queue: MQueue,prompt_queue: Queue):
    prompt_lens = []
    while not stop_event.is_set():
        while not prompt_queue.empty() and not engine.has_unfinished_requests(
            ):
            prompt, sampling_params, prompt_len = prompt_queue.get()
            prompt_lens.append(prompt_len)
            request_id = str(hashlib.md5(prompt.encode('utf-8')).hexdigest())
            engine.add_request(request_id, prompt, sampling_params)
        request_outputs = engine.step()
        for request_output in request_outputs:
            if request_output.finished:
                result_queue.put(time.time())

    


def main(
    max_token_num: int,
    batch_size: int,
    result_queue: MQueue,
    policy: str = "fcfs",
    preemption_mode: str = "swap",
    strategy: str = "full",
    prefill_mode: str = "vertical",
):
    """Main function that sets up and runs the prompt processing."""
    parser = argparse.ArgumentParser(
        description="Demo on using the LLMEngine class directly"
    )

    parser = EngineArgs.add_cli_args(parser)
    args: argparse.Namespace = parser.parse_args()
    args.model = "meta-llama/Llama-2-13b-chat-hf"
    args.swap_space = 16
    args.max_num_seqs = batch_size
    args.scheduler_policy = policy
    args.default_preemption_mode = preemption_mode
    args.max_model_len=16384
    args.enable_chunked_prefill = True
    args.max_num_batched_tokens = max_token_num
    try:
        engine = initialize_engine(args)
    except Exception as e:
        traceback.print_exc()
        print(e)
    print(
        f"start strategy: {strategy}, prefill_mode: {prefill_mode}, policy is {policy}"
    )
    prompts_queue = Queue()
    stop_event = mp.Event()
    executor = ThreadPoolExecutor(max_workers=1)
    executor.submit(
            test_prefilling_time,
            engine,
            stop_event,
            result_queue,
            prompts_queue,
    )
    print("put requests")
    put_requests(stop_event, prompts_queue, result_queue)


if __name__ == "__main__":
    test_type = "test_prefilling_time"
    rerun = True
    with mp.Manager() as manager:
        result_queue = manager.Queue()
        max_token_nums = [65536]
        batch_sizes = [1]
        total_iter_result, total_request_result = Utils.load_tmp_result(
            test_type, BASE_DIR
        )
        preemption_mode = "swap"
        policies = ["fcfs"]
        strategies = ["full"]
        # If prefill mode is horizonal, the sequences length is equals to the token nums, otherwise, the batch size equals to the token nums  # noqa: E501
        prefill_modes = ["horizonal"]
        for strategy in strategies:
            for batch_size in batch_sizes:
                for prefill_mode in prefill_modes:
                    for policy in policies:
                        for max_token_num in max_token_nums:
                            try:
                                with ProcessPoolExecutor(max_workers=2) as executor:
                                    executor.submit(
                                        main,
                                        max_token_num=max_token_num,
                                        batch_size=batch_size,
                                        result_queue=result_queue,
                                        policy=policy,
                                        preemption_mode=preemption_mode,
                                        strategy=strategy,
                                        prefill_mode=prefill_mode,
                                    )
                                    executor.shutdown(wait=True)
                            except Exception as e:
                                traceback.print_exc()
                                print(e)
