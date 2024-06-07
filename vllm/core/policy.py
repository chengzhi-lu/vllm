from collections import deque
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Deque
import numpy as np

from vllm.sequence import SequenceGroup
import random
from torch.nn import functional as F
import torch


class AgentModel:
    def __init__(self, model_name: str = "JackFram/llama-160m"):
        self.model, self.tokenizer, self.eos_token_id = self._init_model(
            model_name
        )

    def _init_model(self, model_name):
        model = AutoModelForCausalLM.from_pretrained(model_name)

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        eos_token_id = tokenizer.eos_token_id
        return model, tokenizer, eos_token_id

    def generate(self, input_ids):
        # Get the model output
        with torch.no_grad():
            output = self.model(input_ids, return_dict=True)
        logits = output["logits"]
        probs = F.softmax(logits, dim=-1)
        next_token_probs = probs[:, -1, self.eos_token_id]
        eos_probs = float(torch.sum(next_token_probs))
        return eos_probs

    def __call__(self, input_ids):
        return self.generate(input_ids)


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
            )
        )


class FCFS(Policy):
    def get_priority(
        self,
        now: float,
        seq_group: SequenceGroup,
    ) -> float:
        return now - seq_group.metrics.arrival_time


class SMLFQ(Policy):
    MLFQ = {}

    def get_priority(
        self,
        now: float,
        seq_group: SequenceGroup,
    ) -> float:
        pass


class InferSchedule(Policy):
    def __init__(self):
        self.agent_model = AgentModel()

    # maximize the number of tokens in the queue while ensuring the sequences with higher probability to finish first.
    def get_priority(self, now: float, seq_group: SequenceGroup) -> float:
        seqs = seq_group.get_seqs()
        input_tokens = []
        for seq in seqs:
            input_tokens.append(seq.get_token_ids())
        eos_probability = self.agent_model(input_tokens)
        return eos_probability


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


class BlockFullPolicy(Policy):
    def get_priority(
        self,
        now: float,
        seq_group: SequenceGroup,
    ) -> float:
        empty_slots = 0
        for _, seq in seq_group.seqs_dict.items():
            for block in seq.logical_token_blocks:
                empty_slots += block.get_num_empty_slots()
        priority = (empty_slots + seq.block_size - 1) % seq.block_size
        return priority


class PolicyFactory:
    _POLICY_REGISTRY = {
        "fcfs": FCFS,
        "utf": UncomputedTokensFirst,
        "random": Random,
        "wtf": WaitingTimeFirst,
        "stf": ShortestTokensFirst,
        "ltf": LongestTokensFirst,
        "bff": BlockFullPolicy,
        "infer": InferSchedule,
    }

    @classmethod
    def get_policy(cls, policy_name: str, **kwargs) -> Policy:
        return cls._POLICY_REGISTRY[policy_name](**kwargs)
