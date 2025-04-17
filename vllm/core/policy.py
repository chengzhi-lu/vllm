from collections import deque
import time
from dataclasses import dataclass
from typing import Deque


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




class TFITTradeoff(Policy):
    def _get_running_priority(self, seq_group: SequenceGroup, policy_info: PolicyInfo):
        if seq_group.priority_rate == 1:
            all_eos_token_prob_diff = []
            for seq in seq_group.seqs_dict.values():
                all_eos_token_prob_diff.append(seq.get_eos_token_prob_diff())
            seq_group.priority_rate = max(all_eos_token_prob_diff)
        priority = (seq_group.priority_rate)/((seq_group.seq_len)/seq_group.max_length)

        return priority

    def _get_swapped_priority(self, seq_group: SequenceGroup, policy_info: PolicyInfo):
        # waiting_time = max(policy_info.now - seq_group.get_last_execute_time(), 0.001)
        waiting_time = seq_group.metrics.waiting_iter_nums
        if seq_group.priority_rate < 1: 
            tmp_priority_rate = seq_group.priority_rate 
            priority = (waiting_time+seq_group.seq_len) / seq_group.max_length* tmp_priority_rate
        elif not seq_group.is_prefill():
            priority = (waiting_time+seq_group.seq_len) / seq_group.max_length
        else:
            priority = (waiting_time) /(seq_group.max_length)
        return priority
        

    def _get_waiting_priority(self, seq_group: SequenceGroup, policy_info: PolicyInfo):
        waiting_time = policy_info.now - seq_group.get_last_execute_time()
        priority = waiting_time/seq_group.seq_len
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
        elif queue_type == "swapped" or queue_type == "waiting":
            priority = self._get_swapped_priority(seq_group, policy_info)
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
        priority = -decode_length if decode_length > 0 else  now -seq_group.metrics.arrival_time
        return priority 


class PolicyFactory:
    _POLICY_REGISTRY = {
        "fcfs": FCFS,
        "random": Random,
        "sjf": ShortJobFirst,
        "srjf": ShortRemainJobFirst,
        "las": LeastAttainedSvr,
        "sjmlfq": SkipJoinMLFQ,
        "tfittradeoff": TFITTradeoff,
    }

    @classmethod
    def get_policy(cls, policy_name: str, **kwargs) -> Policy:
        return cls._POLICY_REGISTRY[policy_name](**kwargs)
