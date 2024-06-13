from collections import deque
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Deque, Dict
import time

from vllm.sequence import SequenceGroup
import random
from torch.nn import functional as F
import torch


class AgentModel:

    def __init__(self, model_name: str = "JackFram/llama-160m"):
        self.model, self.tokenizer, self.eos_token_id = self._init_model(
            model_name)

    def _init_model(self, model_name):
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.to("cuda:0")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        eos_token_id = tokenizer.eos_token_id
        return model, tokenizer, eos_token_id

    def generate(self, input_ids):
        # Get the model output
        input_ids = input_ids.to(self.model.device)
        # print(input_ids)
        with torch.no_grad():
            output = self.model(input_ids, return_dict=True)
        logits = output["logits"]
        probs = F.softmax(logits, dim=-1)
        # print(probs)
        next_token_probs = list(probs[:, -1,
                                      self.eos_token_id].to("cpu").numpy())
        # print(next_token_probs)
        # eos_probs = torch.sum(next_token_probs, dim=0)
        # print(eos_probs)
        return next_token_probs

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
            ))


class FCFS(Policy):

    def get_priority(
        self,
        now: float,
        seq_group: SequenceGroup,
    ) -> float:
        return now - seq_group.metrics.arrival_time


class InferScheduleAgentModel(Policy):

    def __init__(self):
        self.agent_model = AgentModel()

    # maximize the number of tokens in the queue while ensuring the sequences with higher probability to finish first.
    def get_sorted_seq_group(
            self, now: float,
            seq_groups: Deque[SequenceGroup]) -> Deque[SequenceGroup]:
        st = time.time()
        new_seq_groups = [seq for seq in seq_groups]
        input_tokens = []
        for seq_group in new_seq_groups:
            seqs = seq_group.get_seqs()
            for seq in seqs:
                input_tokens.append(seq.get_token_ids())
        input_tokens = torch.tensor(input_tokens)
        et = time.time()
        print("Get input tokens time: ", et - st)
        st = time.time()
        if len(input_tokens) == 0:
            return deque(new_seq_groups)
        eos_probabilities = self.agent_model(input_tokens)
        et = time.time()
        print("Agent model time: ", et - st)
        st = time.time()
        seq_groups_dict = {
            k: v
            for k, v in zip(new_seq_groups, eos_probabilities)
        }
        # sort seq_groups by eos probability
        sorted_seq_groups = sorted(seq_groups_dict.items(),
                                   key=lambda x: x[1],
                                   reverse=True)
        seq_groups = deque([i[0] for i in sorted_seq_groups])
        et = time.time()
        print("parse result time: ", et - st)
        return deque(new_seq_groups)

    def sort_by_priority(
        self,
        now: float,
        seq_groups: Deque[SequenceGroup],
    ) -> Deque[SequenceGroup]:
        return self.get_sorted_seq_group(now, seq_groups)


class InferSchedule(Policy):

    def get_priority(
        self,
        now: float,
        seq_group: SequenceGroup,
    ) -> float:
        eos_token_probs = []
        for seq_id, seq in seq_group.seqs_dict.items():
            eos_token_probs.append(seq.get_eos_token_prob())
        priority = max(
            eos_token_probs) + seq_group.metrics.waiting_iter_nums**2
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
