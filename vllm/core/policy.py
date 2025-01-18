from collections import deque
from os import wait
import time
from dataclasses import dataclass
import math
from typing import Deque

import numpy as np

from vllm.sequence import SequenceGroup
import random


@dataclass
class PolicyInfo:
    waiting_queue_size: int = 0
    running_queue_size: int = 0
    swapped_queue_size: int = 0
    now: float = 0.0


class Policy:
    def get_priority(
        self,
        now: float,
        seq_group: SequenceGroup,
    ) -> float:
        raise NotImplementedError

    def got_priority(
        self,
        seq_group: SequenceGroup,
        queue_type: str,
        policy_info: PolicyInfo,
    ):
        pass

    def sort_by_priority(
        self,
        now: float,
        seq_groups: Deque[SequenceGroup],
    ) -> Deque[SequenceGroup]:
        return deque(
            sorted(
                seq_groups,
                key=lambda seq_group: self.get_priority(now, seq_group),
                reverse=True,
            )
        )

    def sorted_by_priority(
        self,
        seq_groups: Deque[SequenceGroup],
        queue_type: str,
        policy_info: PolicyInfo,
    ) -> Deque[SequenceGroup]:
        policy_info.now = time.time()
        return deque(
            sorted(
                seq_groups,
                key=lambda seq_group: self.got_priority(seq_group, queue_type=queue_type, policy_info=policy_info),
                reverse=True,
            )
        )


class FCFS(Policy):
    def get_priority(
        self,
        now: float,
        seq_group: SequenceGroup,
    ) -> float:
        return now - seq_group.metrics.arrival_time


class MLFQ(Policy):
    def get_priority(
        self,
        now: float,
        seq_group: SequenceGroup,
    ) -> float:
        raise NotImplementedError


class SkipJoinMLFQ(Policy):
    def __init__(self, quantum_ratio=2, starve_limit=2):
        self.quantum_ratio = quantum_ratio  # Q_i/Q_{i-1}
        self.starve_limit = 100  # change to iter num
        self.min_quantum = 30  # quantum of Q_1

    def get_highest_priority(self, first_iteration_time):
        priority_level = 1  # the highest priority
        quantum = self.min_quantum  # the minimum quantum

        while quantum <= first_iteration_time:
            priority_level += 1
            quantum *= self.quantum_ratio

        return priority_level

    def get_priority(self, now: float, seq_group: SequenceGroup) -> float:
        input_length = len(seq_group.prompt_token_ids)

        if seq_group.current_priority is None:  # Have been assigned with a priority?
            seq_group.current_priority = self.get_highest_priority(input_length)
        else:
            if seq_group.metrics.first_scheduled_time is None:   
                seq_group.current_priority = 1
                seq_group.promoted = 1  
            elif (
                now - seq_group.metrics.first_scheduled_time
                > (2 ** (seq_group.current_priority - 1)) * self.min_quantum
                and not seq_group.promoted
            ):
                seq_group.current_priority += 1
            elif seq_group.metrics.waiting_iter_nums >= self.starve_limit:
                seq_group.current_priority = 1  # Promote to highest priority (Q1)
                seq_group.promoted = 1  # has been promoted to the Q1
        

        return -seq_group.current_priority  # higher value means higher priority


class TFTLatencyTrade(Policy):
    def get_gittins_index(self, seq_group: SequenceGroup, eos_probs: float, decoding_length: int):
        # gittins index is the probability of the job ending in the next interval
        # divided by the expected remaining length of the job.
        # Optimization for the request-level latency and ttft

        n = 15
        value = 1 - eos_probs
        eos_probs_in_next_interval = 1 - value**15
        expect_remaining_length = value * ((1 + n * value ** (n + 1) - (n + 1) * (value**n)) / ((1 - value) ** 2))
        gittins_index = eos_probs_in_next_interval / expect_remaining_length
        waiting_percent = seq_group.metrics.waiting_iter_nums**2 * math.sqrt(decoding_length)
        priority = gittins_index * (1 + waiting_percent)
        return priority

    def get_priority(
        self,
        now: float,
        seq_group: SequenceGroup,
    ) -> float:
        eos_token_probs = []
        decoding_length = 0
        # token_blocks = seq_group.total_token_block_size
        for _, seq in seq_group.seqs_dict.items():
            eos_token_probs.extend(seq.get_eos_token_prob())
            decoding_length += seq.get_output_len()
        max_eos_token_prob = max(eos_token_probs)
        if max_eos_token_prob == -1000.0:
            priority = 2000 - seq_group.seq_len
        else:
            probs = math.exp(
                max_eos_token_prob
            )  # short job may have high eos prob. however, this value is too small to be considered.
            priority = self.get_gittins_index(seq_group, probs, decoding_length)
        return priority


class TFITTradeoff(Policy):
    def _get_running_priority(self, seq_group: SequenceGroup, policy_info: PolicyInfo):
        if seq_group.priority_rate == 1:
            all_eos_token_prob_diff = []
            for seq in seq_group.seqs_dict.values():
                all_eos_token_prob_diff.append(seq.get_eos_token_prob_diff())
            seq_group.priority_rate = max(all_eos_token_prob_diff)
        priority = abs((seq_group.priority_rate)**2  -  (seq_group.seq_len / seq_group.max_length)**2)

        return priority

    def _get_swapped_priority(self, seq_group: SequenceGroup, policy_info: PolicyInfo):
        # if policy_info.swapped_queue_size + policy_info.running_queue_size > policy_info.waiting_queue_size:
        #     priority = seq_group.priority_rate * seq_group.metrics.waiting_iter_nums * seq_group.seq_len
        # else:
        # decode_length = sum(seq.get_output_len() for seq in seq_group.seqs_dict.values())
        waiting_time = policy_info.now - seq_group.get_last_execute_time()
        # priority= -waiting_time*seq_group.seq_len
        # priority = -(seq_group.metrics.waiting_iter_nums+1) *seq_group.seq_len
        # priority = -(seq_group.metrics.waiting_iter_nums+1) / (1-seq_group.seq_len/seq_group.max_length)
        # priority=  waiting_time + np.log(seq_group.max_length/seq_group.seq_len)
        priority = waiting_time/seq_group.seq_len
        # priority = waiting_time+1/seq_group.seq_len
        return priority

    def _get_waiting_priority(self, seq_group: SequenceGroup, policy_info: PolicyInfo):
        # if policy_info.swapped_queue_size + policy_info.running_queue_size > policy_info.waiting_queue_size:
        #     priority = -seq_group.metrics.waiting_iter_nums * seq_group.seq_len
        # else:
        # priority = -(seq_group.metrics.waiting_iter_nums+1) *seq_group.seq_len
        # priority= (seq_group.metrics.waiting_iter_nums+1) + np.log(seq_group.max_length/seq_group.seq_len)
        # priority = -(seq_group.metrics.waiting_iter_nums+1) / (1-seq_group.seq_len/seq_group.max_length)
        waiting_time = policy_info.now - seq_group.get_last_execute_time()
        # priority= -waiting_time*seq_group.seq_len
        # priority=  waiting_time + np.log(seq_group.max_length/seq_group.seq_len)
        priority = waiting_time/seq_group.seq_len
        # priority = waiting_time+1/seq_group.seq_len
        return priority

    def got_priority(
        self,
        seq_group: SequenceGroup,
        queue_type: str,
        policy_info: PolicyInfo,
    ) -> float:
        policy_info.now = time.time()
        if queue_type == "running":
            priority = self._get_running_priority(seq_group, policy_info)
        elif queue_type == "swapped":
            priority = self._get_swapped_priority(seq_group, policy_info)
        elif queue_type == "waiting":
            priority = self._get_waiting_priority(seq_group, policy_info)
        else:
            raise ValueError("Invalid queue type")
        return priority

    def get_priority(
        self,
        now: float,
        seq_group: SequenceGroup,
    ) -> float:
        return now - seq_group.metrics.arrival_time


class Random(Policy):
    def get_priority(
        self,
        now: float,
        seq_group: SequenceGroup,
    ) -> float:
        return random.random()


class UncomputedTokensFirst(Policy):
    def get_priority(
        self,
        now: float,
        seq_group: SequenceGroup,
    ) -> float:
        return seq_group.get_num_uncomputed_tokens()


class WaitingTimeFirst(Policy):
    def get_priority(
        self,
        now: float,
        seq_group: SequenceGroup,
    ) -> float:
        return seq_group.metrics.waiting_iter_nums


class ShortRemainJobFirst(Policy):
    def get_priority(
        self,
        now: float,
        seq_group: SequenceGroup,
    ) -> float:
        total_output_lens = sum([seq.get_output_len() for seq in seq_group.get_seqs()])
        priority = -(seq_group.max_length - total_output_lens)
        return priority


class ShortJobFirst(Policy):
    def get_priority(
        self,
        now: float,
        seq_group: SequenceGroup,
    ) -> float:
        priority = -seq_group.max_length
        return priority


class LeastAttainedSvr(Policy):
    def get_priority(
        self,
        now: float,
        seq_group: SequenceGroup,
    ) -> float:
        decode_length = sum(seq.get_output_len() for seq in seq_group.seqs_dict.values())
        # priority = -seq_group.seq_len
        priority = now - seq_group.metrics.arrival_time if decode_length == 0 else -decode_length
        return priority


class LongJobFirst(Policy):
    def get_priority(
        self,
        now: float,
        seq_group: SequenceGroup,
    ) -> float:
        tokens = 0
        waiting_iter_nums = seq_group.metrics.waiting_iter_nums
        for seq_id, seq in seq_group.seqs_dict.items():
            tokens += seq.get_len()
        priority = tokens - waiting_iter_nums * waiting_iter_nums
        return priority


class PolicyFactory:
    _POLICY_REGISTRY = {
        "fcfs": FCFS,
        "utf": UncomputedTokensFirst,
        "random": Random,
        "wtf": WaitingTimeFirst,
        "sjf": ShortJobFirst,
        "srjf": ShortRemainJobFirst,
        "las": LeastAttainedSvr,
        "ljf": LongJobFirst,
        "infer": TFTLatencyTrade,
        "sjmlfq": SkipJoinMLFQ,
        "inferpreempt": TFTLatencyTrade,
        "tfittradeoff": TFITTradeoff,
    }

    @classmethod
    def get_policy(cls, policy_name: str, **kwargs) -> Policy:
        return cls._POLICY_REGISTRY[policy_name](**kwargs)
