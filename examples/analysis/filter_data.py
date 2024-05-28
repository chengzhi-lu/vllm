import random
import json
from vllm.transformers_utils.tokenizer import get_tokenizer
from vllm import EngineArgs
import argparse


def save_seq_to_file(selected_seq):
    import json

    with open("selected_seq_new.json", "w") as f:
        json.dump(selected_seq, f)


def filter_data():
    dataset_path = (
        "/root/vllm/dataset/ShareGPT_V3_unfiltered_cleaned_split.json"
    )
    parser = argparse.ArgumentParser(
        description="Demo on using the LLMEngine class directly"
    )
    parser = EngineArgs.add_cli_args(parser)
    args: argparse.Namespace = parser.parse_args()
    args.model = "meta-llama/Llama-2-13b-hf"
    tokenizer_id = args.tokenizer if args.tokenizer is not None else args.model
    tokenizer = get_tokenizer(
        tokenizer_id, trust_remote_code=args.trust_remote_code
    )
    selected_seqs = {}
    all_need_seq = [i for i in range(1, 2049)]

    with open(dataset_path) as f:
        dataset = json.load(f)
        # Filter out the conversations with less than 2 turns.
        dataset = [data for data in dataset if len(data["conversations"]) >= 2]
        # Only keep the first two turns of each conversation.
        dataset = [
            (
                data["conversations"][0]["value"],
                data["conversations"][1]["value"],
            )
            for data in dataset
        ]

        # Shuffle the dataset.
        random.seed(1)
        random.shuffle(dataset)
        for i in range(len(dataset)):
            selected_seqs_keys = list(selected_seqs.keys())
            if set(selected_seqs_keys) == set(all_need_seq):
                break

            # Tokenize the prompts and completions.
            prompt = dataset[i][0]
            prompt_token_ids = tokenizer(prompt).input_ids
            prompt_len = len(prompt_token_ids)
            if prompt_len not in selected_seqs and prompt_len in all_need_seq:
                selected_seqs[prompt_len] = prompt
        print(max(selected_seqs.keys()))
        print(
            set([i for i in range(1, 2049)]).difference(
                set(selected_seqs.keys())
            )
        )
        save_seq_to_file(selected_seqs)


filter_data()
