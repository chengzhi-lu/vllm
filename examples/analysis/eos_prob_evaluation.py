import argparse
from typing import List, Tuple, Dict

from vllm import EngineArgs, LLMEngine, SamplingParams
from vllm import RequestOutput, LLMEngine
import pandas as pd
import os
import numpy as np
from utils import Utils
from rich import print
from rich import pretty
from rich.progress import track
import traceback


pretty.install()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_requests() -> List[Tuple[str, SamplingParams, int]]:
    init_seq = []
    saved_seq = Utils.load_seq_from_file(BASE_DIR, "seq_data",
                                         "selected_seq.json")
    for p_len in saved_seq:
        prompt_len = int(p_len)
        prompt = saved_seq[p_len]
        init_seq.append((
            prompt,
            SamplingParams(
                temperature=0.0,
                repetition_penalty=2,
                logprobs=1,
                min_tokens=1,
                max_tokens=2000,
            ),
            prompt_len,
        ))

    return init_seq



def initialize_engine(args: argparse.Namespace) -> LLMEngine:
    """Initialize the LLMEngine from the command line arguments."""
    engine_args = EngineArgs.from_cli_args(args)
    return LLMEngine.from_engine_args(engine_args)


def parse_batched_result(request_outputs: List[RequestOutput]):
    _results = []
    for request_output in request_outputs:
        request_id = request_output.request_id
        prompt_len = len(request_output.prompt_token_ids)
        output = request_output.outputs[0]
        output_len = len(output.token_ids)
        if output_len == 0:
            continue
        log_prob = np.exp(output.logprobs[-1][2].logprob)
        rank = request_output.outputs[0].logprobs[-1][2].rank
        _results.append([request_id,prompt_len, output_len, log_prob, rank])
    return _results


def main(
    max_token_num: int,
    batch_size: int,
    policy: str = "fcfs",
    preemption_mode: str = "swap",
):
    """Main function that sets up and runs the prompt processing."""
    parser = argparse.ArgumentParser(
        description="Demo on using the LLMEngine class directly")

    parser = EngineArgs.add_cli_args(parser)
    args: argparse.Namespace = parser.parse_args()
    args.model = "meta-llama/Llama-2-13b-chat-hf"
    args.swap_space = 16
    args.max_num_seqs = batch_size
    args.scheduler_policy = policy
    args.default_preemption_mode = preemption_mode
    args.enable_chunked_prefill = True
    args.max_num_batched_tokens = max_token_num
    try:
        all_inputs = get_requests()
        engine = initialize_engine(args)
    except Exception as e:
        traceback.print_exc()
        print(e)
    eos_result = [] 
    # split inputs into batches with batch_size=16
    split_batch_size = 64
    seqs = [all_inputs[i:i+split_batch_size] for i in range(0, len(all_inputs), split_batch_size)]


    for input_seqs in track(seqs, description="Predicting eos position..."):
        for seq in input_seqs:
            prompt, sampling_params, prompt_len = seq
            engine.add_request(
                request_id=all_inputs.index(seq),
                inputs=prompt,
                params=sampling_params,
            )
        while engine.has_unfinished_requests():
            request_outputs: List[RequestOutput]=engine.step()
            _result = parse_batched_result(request_outputs)
            eos_result.extend(_result) 
        result_df = pd.DataFrame(eos_result, columns=["request_id", "prompt_len", "token_num", "eos_prob", "eos_token_rank"])
        result_df.to_csv(os.path.join(BASE_DIR, "eos_prob_result.csv"), index=False)
    return 
        

def main_test(max_token_num: int,
    batch_size: int,
    policy: str = "fcfs",
    preemption_mode: str = "swap"):
    """Main function that sets up and runs the prompt processing."""
    parser = argparse.ArgumentParser(
        description="Demo on using the LLMEngine class directly")

    parser = EngineArgs.add_cli_args(parser)
    args: argparse.Namespace = parser.parse_args()
    args.model = "meta-llama/Llama-2-13b-chat-hf"
    args.swap_space = 16
    args.max_num_seqs = batch_size
    args.scheduler_policy = policy
    args.default_preemption_mode = preemption_mode
    args.enable_chunked_prefill = True
    args.max_num_batched_tokens = max_token_num
    try:
        all_inputs = get_requests()
        engine = initialize_engine(args)
    except Exception as e:
        traceback.print_exc()
        print(e)
    for i in range(10):
        prompt, sampling_params, prompt_len = all_inputs[0]
        engine.add_request(
                    request_id=0,
                    inputs=prompt,
                    params=sampling_params,
                ) 
        while engine.has_unfinished_requests():
            request_outputs: List[RequestOutput]=engine.step()
        print(len(request_outputs[0].outputs[0].token_ids)+ prompt_len)

    




if __name__ == "__main__":
    test_type = "infer_schedule_policy_test"
    rerun = True
    max_token_nums = [1912]
    batch_sizes = [128]
    total_iter_result, total_request_result = Utils.load_tmp_result(
        test_type, BASE_DIR)
    preemption_mode = "swap"
    policies = ["fcfs"]
    # If prefill mode is horizonal, the sequences length is equals to the token nums, otherwise, the batch size equals to the token nums  # noqa: E501
    for batch_size in batch_sizes:
        for policy in policies:
            for max_token_num in max_token_nums:
                try:
                    main(
                            max_token_num=max_token_num,
                            batch_size=batch_size,
                            policy=policy,
                            preemption_mode=preemption_mode,
                        )
                    # main_test(
                    #     max_token_num=max_token_num,
                    #     batch_size=batch_size,
                    #     policy=policy,
                    #     preemption_mode=preemption_mode,
                    # )
                except Exception as e:
                    traceback.print_exc()
                    print(e)

