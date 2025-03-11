from typing import List
import numpy as np


class BatchSovler:
    def __init__(self):
        self.decode_time_params = np.array([0, 0, 0])
        self.prefill_time_params = np.array([0, 0, 0])
        self.sample_time_params = np.array([0,0])
        

    def _get_params(self, parallel_type: str, pipeline_parallel_size: int=0):
        if parallel_type == "pp":
            if pipeline_parallel_size == 4:
                self.decode_time_params = np.array([1.00, 0.00, 0.00])
                self.prefill_time_params = np.array([0.00, 0.00, 0.00])
                self.sample_time_params = np.array([0.00, 0.00])
            elif pipeline_parallel_size == 8:
                self.decode_time_params = np.array([0.50, 0.00, 0.00])
                self.prefill_time_params = np.array([0.00, 0.00, 0.00])
                self.sample_time_params = np.array([0.00, 0.00])
        elif parallel_type == "tp":
            self.decode_time_params = np.array([2.00, 0.00, 0.00])
            self.prefill_time_params = np.array([0.00, 0.00, 0.00])
            self.sample_time_params = np.array([0.00, 0.00])
        elif parallel_type == "single":
            self.decode_time_params = np.array([3.00, 0.00, 0.00])
            self.prefill_time_params = np.array([1.168e-05, 0.1064, 21.857])
            self.sample_time_params = np.array([0.00, 0.00])
        else:
            self.decode_time_params = np.array([4.00, 0.00, 0.00])
            self.prefill_time_params = np.array([0.00, 0.00, 0.00])
            self.sample_time_params = np.array([0.00, 0.00])

    def get_best_token_limits(self, decode_seqs: List[int]):
        b = len(decode_seqs)
        s = np.sum(decode_seqs)
        decode_time = self.decode_time_params[0] * b + self.decode_time_params[1] * b * s + self.decode_time_params[2]
        sampling_time = self.sample_time_params[0] * (b+1) + self.sample_time_params[1] 
        token_limit = int(-b) + np.sqrt(
            (
                self.prefill_time_params[0] * b * b
                - b * self.prefill_time_params[1]
                + self.prefill_time_params[2]
                + decode_time
                + sampling_time
            )
            / self.prefill_time_params[0]
        )
        if token_limit < 0:
            token_limit = 0
        return token_limit
