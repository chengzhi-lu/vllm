from collections import deque
from time import time
from typing import List
from vllm.core.policy import Policy
from vllm.core.scheduler import SchedulingBudget
from vllm.sequence import SequenceGroup


class InferSchedule:
    def __init__(self, max_seq_num:int, max_token_num:int):
        self.max_seq_num = max_seq_num
        self.max_token_num = max_token_num
        # record sequence occupied token blocks and token size
        self.seq_infos_matrix=[]
    
    def _parse_seq_infos(self, seq_groups: deque):
        for seq_group in seq_groups:
            request_id = seq_group.request_id
            token_block_size= 0
            token_num = 0
            for seq in seq_group.get_seqs():
                token_block_size+=seq.logical_token_block_size 
                token_num+=seq.get_len()
            self.seq_infos_matrix.append([request_id, token_block_size, token_num])
             
        
        
    
    def _selects_running_seqs(self, total_seq_queue: deque, policy: Policy, budget: SchedulingBudget) -> List[SequenceGroup]:
        sorted_seq_queue=policy.sort_by_priority(time.time(),total_seq_queue)
        self._parse_seq_infos(sorted_seq_queue)
        