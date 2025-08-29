import time
from typing import List
import numpy as np
import pandas as pd
import os

from vllm.sequence import SequenceGroup


class BatchSolver:
    def __init__(
        self,
        parallel_type: str,
        pipeline_parallel_size: int,
        model_id: str,
    ):
        self.decode_time_params = np.array([0, 0, 0])
        self.prefill_time_params = np.array([0, 0, 0])
        self.sample_time_params = np.array([0, 0])
        self.profile_dir = "/root/vllm/vllm/core/profile_data/"
        self.profile_result = self._read_params()
        if self.profile_result is None:
            return None
        self.get_profiled_info()
        self._get_params(
            parallel_type=parallel_type,
            pipeline_parallel_size=pipeline_parallel_size,
            model_id=model_id,
        )
        self.max_throughput = {}  # b:tp
        self.last_token_limit = {}  # b:token_limit
        self.total_waiting_time = 0
        self.total_execution_time = 0
        self.decode_seqs = []

    def get_profiled_info(self):
        try:
            if not hasattr(self, "profile_result") or not isinstance(self.profile_result, pd.DataFrame):
                raise ValueError("self.profile_result must be a valid pandas DataFrame.")

            required_columns = {"model_id", "parallel_type", "num_instances"}
            if not required_columns.issubset(self.profile_result.columns):
                missing_cols = required_columns - set(self.profile_result.columns)
                raise KeyError(f"Missing required columns in self.profile_result: {missing_cols}")

            def extract_unique_values(column_name):
                unique_values = self.profile_result[column_name].dropna().unique().tolist()
                return unique_values if unique_values else []

            self.profiled_model_ids = extract_unique_values("model_id")
            self.profiled_parallel_types = extract_unique_values("parallel_type")
            self.profiled_parallel_instances = extract_unique_values("num_instances")
        except (ValueError, KeyError, AttributeError) as e:
            print(f"Error in get_profiled_info: {e}")
            raise

    def _read_params(self):
        data_frames = []
        if self.profile_dir is None or not os.path.exists(self.profile_dir) or not os.path.isdir(self.profile_dir):
            return None
        for file_name in os.listdir(self.profile_dir):
            if file_name.endswith(".csv"):
                try:
                    file_path = os.path.join(self.profile_dir, file_name)
                    tmp_df = pd.read_csv(file_path)
                    if "_" in file_name:
                        stage = file_name.split("_")[0]
                        tmp_df["stage"] = stage
                    else:
                        print(f"Warning: File name '{file_name}' does not match expected format, skipping.")
                        continue
                    data_frames.append(tmp_df)
                except Exception as e:
                    print(f"Error processing file '{file_name}': {e}")

        if not data_frames:
            print("No valid CSV files found in the directory.")
            return pd.DataFrame()

        total_profile_result = pd.concat(data_frames, ignore_index=True)
        total_profile_result = (
            total_profile_result.groupby(["model_id", "parallel_type", "num_instances", "stage"]).mean().reset_index()
        )
        return total_profile_result

    def _get_params(self, parallel_type: str, pipeline_parallel_size: int = 0, model_id: str = ""):
        if not parallel_type or pipeline_parallel_size < 0 or not model_id:
            print("Invalid parameters: parallel_type, pipeline_parallel_size, and model_id must be valid.")
            return

        if (
            parallel_type not in self.profiled_parallel_types
            or pipeline_parallel_size not in self.profiled_parallel_instances
            or model_id not in self.profiled_model_ids
        ):
            print("No profiled data for this combination of parameters")
            return

        def get_profiled_values(stage, columns):
            try:
                filtered_data = self.profile_result[
                    (self.profile_result["parallel_type"] == parallel_type)
                    & (self.profile_result["num_instances"] == pipeline_parallel_size)
                    & (self.profile_result["model_id"] == model_id)
                    & (self.profile_result["stage"] == stage)
                ][columns]
                if filtered_data.empty:
                    raise ValueError(
                        f"No data found for stage '{stage}' with given parameters {parallel_type}, {pipeline_parallel_size}, {model_id}."
                    )
                return filtered_data.iloc[0].tolist()
            except Exception as e:
                print(f"Error retrieving data for stage '{stage}': {e}")
                return None

        decode_columns = ["a", "b", "c"]
        prefill_columns = ["a", "b", "c"]
        sample_columns = ["a", "b"]

        self.decode_time_params = get_profiled_values("decode", decode_columns)
        self.prefill_time_params = get_profiled_values("prefill", prefill_columns)
        self.sample_time_params = get_profiled_values("sample", sample_columns)

        if not all([self.decode_time_params, self.prefill_time_params, self.sample_time_params]):
            print("Some parameters could not be retrieved successfully.")

    def is_opt(self, scheduling_policy: str, seq_group: SequenceGroup) -> bool:
        if np.sum(self.decode_time_params) == 0 or scheduling_policy != "tfittradeoff":
            return True
        now = time.time()
        waiting_time = (
            now - seq_group.get_last_execute_time()
            if seq_group.metrics.time_in_queue
            else now - seq_group.metrics.arrival_time
        )
        if seq_group.is_prefill():
            delta_execution_time = (
                self.prefill_time_params[0] * seq_group.seq_len**2
                + self.prefill_time_params[1] * seq_group.seq_len
                + self.prefill_time_params[2]
            ) 
        else:
            self.decode_seqs.append(seq_group.seq_len)
            delta_execution_time = (
                self.decode_time_params[0]
                + self.decode_time_params[1] * seq_group.seq_len
                + self.decode_time_params[2]
            ) 
        execution_time = delta_execution_time / 1000

        if self.total_execution_time == 0:
            self.total_execution_time = execution_time
            self.total_waiting_time = waiting_time
            return True
        else:
            if waiting_time / execution_time >= self.total_waiting_time / self.total_execution_time:
                self.total_waiting_time += waiting_time
                self.total_execution_time += execution_time
                return True
            else:
                if not seq_group.is_prefill():
                    self.decode_seqs = self.decode_seqs[:-1]
                return False

    def reset_opt(self):
        self.decode_seqs = []
        self.total_waiting_time = 0
        self.total_execution_time = 0

    def get_best_token_limits(self, scheduling_policy: str, decode_seqs: List[int]):
        if len(decode_seqs) == 0 or np.sum(self.decode_time_params) == 0 or scheduling_policy != "tfittradeoff":
            return 0
        b = len(decode_seqs)
        s = np.sum(decode_seqs)
        decode_time = self.decode_time_params[0] * b + self.decode_time_params[1] * b * s + self.decode_time_params[2]
        sampling_time = self.sample_time_params[0] * (b + 1) + self.sample_time_params[1]

        try:
            sqrt_param = (
                b**2
                + (decode_time + sampling_time + self.prefill_time_params[2] - self.prefill_time_params[1] * b)
                / self.prefill_time_params[0]
            )

            if sqrt_param < 0:
                print(f"Square root parameter is negative. b is {b}, s is {s}")
                return 0
            token_limit = int(-b + np.sqrt(sqrt_param))
            predicted_prefill_time = (
                self.prefill_time_params[0] * token_limit**2
                + self.prefill_time_params[1] * token_limit
                + self.prefill_time_params[2]
            )
            max_throughput = (token_limit + b) / (decode_time + sampling_time + predicted_prefill_time)
            if b not in self.max_throughput:
                self.max_throughput[b] = max_throughput
                self.last_token_limit[b] = token_limit
                return token_limit
            if max_throughput > self.max_throughput[b]:
                self.max_throughput[b] = max_throughput
                self.last_token_limit[b] = token_limit
            return self.last_token_limit[b]
        except ZeroDivisionError as e:
            raise ValueError("prefill_time_params[0] cannot be zero.") from e
