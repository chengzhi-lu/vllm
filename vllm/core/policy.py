from collections import deque
import math
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
            if now-seq_group.metrics.first_scheduled_time > (2**(seq_group.current_priority-1))*self.min_quantum and not seq_group.promoted:
                seq_group.current_priority += 1
            elif seq_group.metrics.time_in_queue >= self.starve_limit:
                seq_group.current_priority = 1  # Promote to highest priority (Q1)
                seq_group.promoted = 1 # has been promoted to the Q1

        return -seq_group.current_priority # higher value means higher priority

class TFTLatencyTrade(Policy):

    def get_gittins_index(self,seq_group:SequenceGroup, eos_probs: float, decoding_length: int):
        # gittins index is the probability of the job ending in the next interval 
        # divided by the expected remaining length of the job.
        # Optimization for the request-level latency and ttft

        n=15
        value = 1-eos_probs
        eos_probs_in_next_interval = 1 -value**15
        expect_remaining_length =value*((1+n*value**(n+1)-(n+1)*(value**n))/(((1-value)**2))) 
        gittins_index =eos_probs_in_next_interval / expect_remaining_length
        waiting_percent = \
               seq_group.metrics.waiting_iter_nums**2 * math.sqrt(decoding_length)
        priority = gittins_index*(1+waiting_percent)
        return priority
        
    

    def get_priority(
        self,
        now: float,
        seq_group: SequenceGroup,
    ) -> float:
        eos_token_probs = []
        decoding_length = 0
        seq_length = seq_group.seq_len
        # token_blocks = seq_group.total_token_block_size
        for _, seq in seq_group.seqs_dict.items():
            eos_token_probs.extend(seq.get_eos_token_prob())
            decoding_length += seq.get_output_len()
        max_eos_token_prob = max(eos_token_probs)
        if max_eos_token_prob == -1000.0:
            priority = (2000-seq_group.seq_len)
        else:
            probs = math.exp(max_eos_token_prob) # short job may have high eos prob. however, this value is too small to be considered.
            priority = self.get_gittins_index(seq_group,probs,decoding_length)
        return priority


class TFITTradeoff(Policy):

    def get_waiting_index(self,seq_group:SequenceGroup, eos_probs: float):
        # waiting index is the probability of the job 
        expected_length = seq_group.expected_length
        if expected_length == 0:
            seq_len = seq_group.seq_len
            value = 1-eos_probs
            n=15
            expect_remaining_length = value*((1+n*value**(n+1)-(n+1)*(value**n))/(((1-value)**2)))
            # index = -seq_group.metrics.waiting_iter_nums**2 / math.sqrt(expect_remaining_length)
            seq_group.expected_length = (seq_len+expect_remaining_length)
            expected_length = (seq_len+expect_remaining_length)
        index = -expected_length
        return index 

    def get_priority(
        self,
        now: float,
        seq_group: SequenceGroup,
    ) -> float:
        # eos_token_probs = []
        seq_len = seq_group.seq_len
        max_eos_token_prob = -1000.0
        for _, seq in seq_group.seqs_dict.items():
            tmp_max=max(seq.get_eos_token_prob())
            if tmp_max > max_eos_token_prob:
                max_eos_token_prob = tmp_max
        if max_eos_token_prob == -1000.0:
            priority = -seq_len
        else:
            probs = math.exp(max_eos_token_prob) # short job may have high eos prob. however, this value is too small to be considered.
            # priority = self.get_waiting_index(seq_group, probs)
            expected_length = seq_group.expected_length
            if expected_length == 0:
                value = 1-probs
                n=15
                expect_remaining_length = value*((1+n*value**(n+1)-(n+1)*(value**n))/(((1-value)**2)))
                # index = -seq_group.metrics.waiting_iter_nums**2 / math.sqrt(expect_remaining_length)
                seq_group.expected_length = (seq_len+expect_remaining_length)
                expected_length = (seq_len+expect_remaining_length)
            priority = -expected_length
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


class ShortJobFirst(Policy):

    def get_priority(
        self,
        now: float,
        seq_group: SequenceGroup,
    ) -> float:
        tokens = 0
        for seq_id, seq in seq_group.seqs_dict.items():
            tokens += seq.get_len()
        priority =  - tokens
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
        "ljf": LongJobFirst,
        "infer": TFTLatencyTrade,
        "sjmlfq": SkipJoinMLFQ,
        "inferpreempt": TFTLatencyTrade,
        "tfittradeoff": TFITTradeoff,
    }

    @classmethod
    def get_policy(cls, policy_name: str, **kwargs) -> Policy:
        return cls._POLICY_REGISTRY[policy_name](**kwargs)
