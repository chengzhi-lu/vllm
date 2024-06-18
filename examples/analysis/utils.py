import time
import numpy as np
from typing import List, Dict
from dataclasses import dataclass
from vllm import RequestOutput, LLMEngine
import pandas as pd
import os
from queue import Queue
from multiprocessing import Queue as MQueue
import uuid
import traceback


@dataclass
class RequestMetrics:
    request_id: str
    prompt_length: int
    decode_length: int
    request_start_time: float
    request_end_time: float
    request_latency: float
    batch_size: int
    random_seed: int
    strategy: str
    enable_chunk_prefill: bool
    policy: str
    preemption_mode: str


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
    num_preemption_iter: int
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
    max_token_num: int
    prefill_mode: str


class Utils:

    @staticmethod
    def calc_cv(prompt_lens: List[int]):
        if len(prompt_lens) == 0:
            return 0
        prompt_lens_mean = np.mean(prompt_lens)
        prompt_lens_std = np.std(prompt_lens)
        return prompt_lens_std / prompt_lens_mean

    @staticmethod
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
        max_token_num: int,
        prefill_mode: str,
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
        num_preemption_iter = 0
        request_num_each_round = 0
        for request_output in request_outputs:
            request_num_each_round += 1
            request_id = request_output.request_id
            num_waiting_to_running = request_output.num_running_to_waiting
            num_running_to_waiting = request_output.num_waiting_to_running
            recomputed_token_nums = request_output.recomputed_token_nums
            num_preemption_iter = request_output.num_preemption_iter
            if len(request_output.outputs[0].token_ids) == 0:
                # prefill stage
                prefill_token_nums += request_output.token_chunk_size
                prefill_token_num_each_request.append(
                    (request_id, request_output.token_chunk_size))
                prefill_stage = True
            elif len(request_output.outputs[0].token_ids) == 1:
                prefill_token_nums += request_output.token_chunk_size
                prefill_token_num_each_request.append(
                    (request_id, request_output.token_chunk_size))
                decode_token_nums += 0
                prefill_stage = True
            else:
                decode_token_nums += 1
                decode_stage = True
            if request_output.finished:
                request_metrics[f"{request_id}"].request_end_time = time.time()
                request_metrics[f"{request_id}"].decode_length = len(
                    request_output.outputs[0].token_ids)
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
            cvs=Utils.calc_cv(prompt_lens),
            batch_size=batch_size,
            request_num=len(request_metrics),
            random_seed=request_round,
            strategy=strategy,
            enable_chunk_prefill=enable_chunk_prefill,
            policy=policy,
            num_running_to_waiting=num_running_to_waiting,
            num_waiting_to_running=num_waiting_to_running,
            recomputed_token_nums=recomputed_token_nums,
            num_preemption_iter=num_preemption_iter,
            current_stage=current_stage,
            wasted_block_sizes=wasted_block_sizes,
            total_block_sizes=total_block_sizes,
            repeat_time=repeat_time,
            max_token_num=max_token_num,
            prefill_mode=prefill_mode,
        )

        return request_metrics, iter_metric

    @staticmethod
    def convert_request_metrics(request_metrics: Dict[str, RequestMetrics]):
        request_metrics = request_metrics.values()
        request_metrics = list(request_metrics)
        df = pd.DataFrame([vars(m) for m in request_metrics])
        df.set_axis(RequestMetrics.__dataclass_fields__, axis=1)
        return df

    @staticmethod
    def convert_iter_metrics(iter_metrics: List[IterMetrics]):
        df = pd.DataFrame([vars(m) for m in iter_metrics])
        df.set_axis(IterMetrics.__dataclass_fields__, axis=1)
        return df

    @staticmethod
    def save_result(
        total_iter_result,
        total_request_result,
        enable_chunk_prefill,
        test_type,
        rerun,
        BASE_DIR,
    ):
        chunks = ""
        if not enable_chunk_prefill:
            chunks = "not_chunks_"

        dir_request_result_tmp = os.path.join(BASE_DIR, "data/tmp")

        if not os.path.exists(dir_request_result_tmp):
            os.makedirs(dir_request_result_tmp)

        request_result_tmp_path = os.path.join(
            dir_request_result_tmp, f"tmp_{test_type}_request_level.csv")

        dir_request_result = os.path.join(
            BASE_DIR,
            "data/request_level",
        )

        if not os.path.exists(dir_request_result):
            os.makedirs(dir_request_result)

        request_result_path = os.path.join(
            dir_request_result,
            f"{chunks}request_result_{test_type}_policy.csv",
        )

        iter_result_tmp_path = os.path.join(BASE_DIR, "data/tmp",
                                            f"tmp_{test_type}_iter_level.csv")

        dir_iter_result = os.path.join(
            BASE_DIR,
            "data/iter_level",
        )

        if not os.path.exists(dir_iter_result):
            os.makedirs(dir_iter_result)

        iter_result_path = os.path.join(
            dir_iter_result,
            f"{chunks}iter_result_{test_type}_policy.csv",
        )

        total_request_result.to_csv(request_result_tmp_path, index=False)
        total_iter_result.to_csv(iter_result_tmp_path, index=False)
        if not rerun:
            if os.path.exists(iter_result_path):
                tmp = pd.read_csv(iter_result_path)
                total_iter_result_policy = total_request_result[
                    "policy"].unique()
                total_iter_result_batch_size = total_request_result[
                    "batch_size"].unique()
                # remove all rows that have the same policy and batch_size
                tmp = tmp[
                    ~(tmp["policy"].isin(total_iter_result_policy))
                    | ~(tmp["batch_size"].isin(total_iter_result_batch_size))]
                total_iter_result = pd.concat([tmp, total_iter_result])
            else:
                total_iter_result.to_csv(iter_result_path, index=False)
            if os.path.exists(request_result_path):
                tmp = pd.read_csv(request_result_path)
                total_request_result_policy = total_request_result[
                    "policy"].unique()
                total_request_result_batch_size = total_request_result[
                    "batch_size"].unique()
                tmp = tmp[~(tmp["policy"].isin(total_request_result_policy))
                          | ~(tmp["batch_size"].
                              isin(total_request_result_batch_size))]
                total_request_result = pd.concat([tmp, total_request_result])
            else:
                total_request_result.to_csv(
                    request_result_path,
                    index=False,
                )
        else:
            total_request_result.to_csv(
                request_result_path,
                index=False,
            )
            total_iter_result.to_csv(iter_result_path, index=False)
        os.remove(request_result_tmp_path)
        os.remove(iter_result_tmp_path)

    @staticmethod
    def load_tmp_result(test_type, BASE_DIR):
        iter_result_path = os.path.join(BASE_DIR, "data/tmp",
                                        f"tmp_{test_type}_iter_level.csv")
        request_result_path = os.path.join(
            BASE_DIR, "data/tmp", f"tmp_{test_type}_request_level.csv")
        if os.path.exists(iter_result_path) and os.path.exists(
                request_result_path):
            total_iter_result = pd.read_csv(iter_result_path)
            total_request_result = pd.read_csv(request_result_path)
        else:
            total_iter_result = pd.DataFrame()
            total_request_result = pd.DataFrame()
        return total_iter_result, total_request_result

    @staticmethod
    def save_tmp_result(total_iter_result, total_request_result, test_type,
                        BASE_DIR):
        path = os.path.join(BASE_DIR, "data/tmp")

        if not os.path.exists(path):
            os.makedirs(path)

        iter_result_path = os.path.join(path,
                                        f"tmp_{test_type}_iter_level.csv")
        request_result_path = os.path.join(
            path, f"tmp_{test_type}_request_level.csv")
        total_iter_result.to_csv(iter_result_path, index=False)
        total_request_result.to_csv(request_result_path, index=False)

    @staticmethod
    def save_seq_to_file(saved_seq, BASE_DIR):
        import json

        saved_seq_file = os.path.join(BASE_DIR, "saved_seq.json")
        with open(saved_seq_file, "w") as f:
            json.dump(saved_seq, f)

    def load_seq_from_file(BASE_DIR, DATA_DIR, seq_file_name="saved_seq.json"):
        import json

        saved_seq_file = os.path.join(BASE_DIR, DATA_DIR, seq_file_name)
        if os.path.exists(saved_seq_file):
            with open(saved_seq_file, "r") as f:
                saved_seq = json.load(f)
            return saved_seq
        else:
            saved_seq = {"init": [], "prefill": []}
            return saved_seq

    @staticmethod
    def update_request_end_time(
        request_outputs: List[RequestOutput],
        request_metrics: Dict[str, RequestMetrics],
    ):
        for request_output in request_outputs:
            request_metrics[
                request_output.
                request_id].request_end_time = request_output.metrics.finished_time
            if request_output.finished:
                request_metrics[request_output.request_id].request_latency = (
                    request_metrics[request_output.request_id].request_end_time
                    - (request_metrics[
                        request_output.request_id].request_start_time))

    @staticmethod
    def process_requests(
        engine: LLMEngine,
        prompts_queue: Queue,
        add_new_request_notice: Queue,
        strategy: str,
        result_queue: MQueue,
        batch_size: int,
        enable_chunk_prefill: bool = False,
        policy: str = "fcfs",
        repeat_time: int = 1,
        max_token_num: int = 2048,
        random_seed: int = 10,
        prefill_mode: str = "vertical",
        insert_new_request: bool = False,
        insert_new_request_round: int = -1,
        preemption_mode: str = "recompute",
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
            while not prompts_queue.empty() or engine.has_unfinished_requests(
            ):
                while not prompts_queue.empty():
                    prompt, sampling_params, prompt_len = prompts_queue.get()
                    prompt_lens.append(prompt_len)
                    if prompt_len in prompt_lens:
                        request_id = str(
                            uuid.uuid5(namespace,
                                       prompt + str(duplicate_prompt)))
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
                        request_latency=-1,
                        batch_size=batch_size,
                        random_seed=random_seed,
                        enable_chunk_prefill=enable_chunk_prefill,
                        strategy=strategy,
                        policy=policy,
                        preemption_mode=preemption_mode,
                    )
                request_round = request_round + 1
                st = time.time()
                request_outputs: List[RequestOutput] = engine.step()
                et = time.time()
                Utils.update_request_end_time(request_outputs, request_metrics)
                # try:
                #     if engine.log_stats:
                #         if engine.stat_logger._local_interval_elapsed(engine.stats.now):
                #             print(f"The iteration time: {et - st}")
                #     print(engine.stats.now)
                # except Exception as e:
                #     print("error", e)

                if (insert_new_request
                        and request_round == insert_new_request_round):
                    add_new_request_notice.put(1)
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
                    prefill_mode=prefill_mode,
                )
                iter_metrics.append(iter_metric)

            request_result_metric = Utils.convert_request_metrics(
                request_metrics)
            iter_result_metric = Utils.convert_iter_metrics(iter_metrics)
        except Exception as e:
            traceback.print_exc()
            print(e)
        result_queue.put((iter_result_metric, request_result_metric))
        return
