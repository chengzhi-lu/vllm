from collections import deque
from typing import Deque
import numpy as np

from vllm.sequence import SequenceGroup
import random


class Policy:

    def get_priority(
        self,
        now: float,
        seq_group: SequenceGroup,
    ) -> float:
        raise NotImplementedError

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
            ))


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
    def __init__(self, quantum_ratio=2, starve_limit=1000):
        self.quantum_ratio = quantum_ratio # Q_i/Q_{i-1}
        self.starve_limit = starve_limit
        self.min_quantum = 1000 # quantum of Q_1

    def get_highest_priority(self, first_iteration_time):
        priority_level = 1 # the highest priority
        quantum = self.min_quantum # the minimum quantum

        while quantum <= first_iteration_time:
            priority_level += 1
            quantum *= self.quantum_ratio

        return priority_level

    def get_priority(self, now: float, seq_group: SequenceGroup) -> float:
        input_length = len(seq_group.seqs_dict)

        # first_token_time = seq_group.metrics.first_token_time # Obtain the first_iteration_time for each job
        arrival_time = seq_group.metrics.arrival_time

        # Assign priority based on first iteration time
        if not seq_group.current_priority: # Have been assigned with a priority?
            seq_group.current_priority = self.get_highest_priority(input_length)
        else:
            if now-seq_group.metrics.first_scheduled_time > (2**(priority-1))*self.min_quantum and not seq_group.promoted:
                seq_group.current_priority += 1
            elif seq_group.metrics.time_in_queue >= self.starve_limit:
                seq_group.current_priority = 1  # Promote to highest priority (Q1)
                seq_group.promoted = 1 # has been promoted to the Q1

        return -seq_group.current_priority # higher value means higher priority

class InferSchedule(Policy):

    def get_gittins_index(self, eos_probs: float):
        # gittins index is the probability of the job ending in the next interval divided by the expected remaining length of the job.
        i = np.arange(1, 14)
        eos_probs_in_next_interval = 1 - np.power((1 - eos_probs), 14)
        expect_remaining_length = np.sum(i * (1 - eos_probs) ** i)
        return eos_probs_in_next_interval / expect_remaining_length 

    def get_penalty(self, decoding_length: int, eos_probs: float):
        # only consider the gittins index is not enough due to the long sequence 
        # may occupy too much GPU memory and block the inference of other sequences.
        # we add a penalty term to the priority to avoid this.
        eos_probs_before_decoding = 1 - np.power((1 - eos_probs), decoding_length)
        

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
        max_eos_token_prob = np.max(eos_token_probs)
        if max_eos_token_prob == -1000.0:
            # priority = len(seq_group.prompt_token_ids)
            if decoding_length == 0:
                priority = (1000-len(seq_group.prompt_token_ids))
            else:
                priority = (1000-decoding_length)
        else:
            probs = np.exp(max_eos_token_prob) # short job may have high eos prob. however, this value is too small to be considered.
            priority = self.get_gittins_index(probs)
            # print(f"Seq group is {seq_group.request_id}, priority is {priority}")
            # probs = mean_eos_token_prob
            # waiting_percent = \
            #     seq_group.metrics.waiting_iter_nums**1.5 / decoding_length
            # priority = probs + waiting_percent
        return priority


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


class ShortestTokensFirst(Policy):

    def get_priority(
        self,
        now: float,
        seq_group: SequenceGroup,
    ) -> float:
        tokens = 0
        waiting_iter_nums = seq_group.metrics.waiting_iter_nums
        for seq_id, seq in seq_group.seqs_dict.items():
            tokens += seq.get_len()
        priority = waiting_iter_nums * waiting_iter_nums - tokens
        return priority


class LongestTokensFirst(Policy):

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
        "stf": ShortestTokensFirst,
        "ltf": LongestTokensFirst,
        "infer": InferSchedule,
    }

    @classmethod
    def get_policy(cls, policy_name: str, **kwargs) -> Policy:
        return cls._POLICY_REGISTRY[policy_name](**kwargs)
