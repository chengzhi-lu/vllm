import random
import json
from vllm.transformers_utils.tokenizer import get_tokenizer
from vllm import EngineArgs
import argparse


def save_seq_to_file(selected_seq, file_name):
    filter_data_path = "/root/vllm/examples/analysis/seq_data"
    with open(f"{filter_data_path}/{file_name}.json", "w") as f:
        json.dump(selected_seq, f)


def filter_data_alpaca(model_name):
    dataset_path = (
        "/root/vllm/dataset/alpaca_data.json"
    )
    parser = argparse.ArgumentParser(
        description="Demo on using the LLMEngine class directly"
    )
    parser = EngineArgs.add_cli_args(parser)
    args: argparse.Namespace = parser.parse_args()
    if model_name == 'llama':
        args.model = "meta-llama/Llama-2-13b-hf"
    elif model_name == 'mistral':
        args.model="mistralai/Mistral-7B-Instruct-v0.1"
    tokenizer_id = args.tokenizer if args.tokenizer is not None else args.model
    tokenizer = get_tokenizer(
        tokenizer_id, trust_remote_code=args.trust_remote_code
    )
    selected_seqs = {}
    # all_need_seq = [i for i in range(1, 2049)]
    dataset=[]
    with open(dataset_path) as f:
        dataset = json.load(f)
        import random
        # Only keep the first two turns of each conversation.
        dataset = [
            {
                "id": f"{random.randint(0,len(dataset))}",
                "conversation":[
                    {
                    "from": "human",
                    "value": data["instruction"] + " "+ data["input"]
                    },
                    {"from": "gpt",
                     "value":data["output"]}
                ]
            }
            for data in dataset
        ]
    with open("test.json", mode='w') as f:
        f.write(json.dumps(dataset))

        # # Shuffle the dataset.
        # random.seed(1)
        # random.shuffle(dataset)
        # count = 0
        # for i in range(len(dataset)):
        #     prompt = dataset[i][0]
        #     prompt_token_ids = tokenizer(prompt).input_ids
        #     prompt_len = len(prompt_token_ids)
        #     selected_seqs[count] = (prompt,prompt_len)
        #     count+=1
        # file_name=f"{model_name}_alpaca"
        # save_seq_to_file(selected_seqs,file_name)

def filter_data_sharegpt(model_name):
    dataset_path = (
        "/root/vllm/dataset/ShareGPT_V3_unfiltered_cleaned_split.json"
    )
    parser = argparse.ArgumentParser(
        description="Demo on using the LLMEngine class directly"
    )
    parser = EngineArgs.add_cli_args(parser)
    args: argparse.Namespace = parser.parse_args()
    if model_name == 'llama':
        args.model = "meta-llama/Llama-2-13b-hf"
    elif model_name == 'mistral':
        args.model="mistralai/Mistral-7B-Instruct-v0.1"
    tokenizer_id = args.tokenizer if args.tokenizer is not None else args.model
    tokenizer = get_tokenizer(
        tokenizer_id, trust_remote_code=args.trust_remote_code
    )
    selected_seqs = {}
    # all_need_seq = [i for i in range(1, 2049)]

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
        count = 0
        for i in range(len(dataset)):
            # selected_seqs_keys = list(selected_seqs.keys())
            # if set(selected_seqs_keys) == set(all_need_seq):
            #     break

            # Tokenize the prompts and completions.
            prompt = dataset[i][0]
            prompt_token_ids = tokenizer(prompt).input_ids
            prompt_len = len(prompt_token_ids)
            selected_seqs[count] = (prompt,prompt_len)
            count = count + 1
        # print(max(selected_seqs.keys()))
        # print(
        #     set([i for i in range(1, 2049)]).difference(
        #         set(selected_seqs.keys())
        #     )
        # )
        file_name=f"{model_name}_sharegpt"
        save_seq_to_file(selected_seqs,file_name)


for model in ["llama"]:
    # filter_data_sharegpt(model)
    filter_data_alpaca(model)