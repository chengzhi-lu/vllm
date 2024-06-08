# use JackFram/llama-68m in the huggingface to predict the eos of the prompt
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    LlamaTokenizer,
)
from typing import List
import json
import random
from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class Model:
    model_name: str
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer


def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = ""
    m = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda:1")
    model = Model(model_name, m, tokenizer)
    return model


def predict(model: Model, inputs: torch.Tensor, eos_token_id: int = 2):
    eos_poss = []
    next_token = inputs[0][-1]
    print(model.model)
    with torch.no_grad():
        while next_token != eos_token_id:
            generate_ids = model.model(
                inputs,
                return_dict=True,
            )
            logits = generate_ids["logits"]
            probs = torch.nn.functional.softmax(logits, dim=-1)
            next_token_probs = probs[:, -1]
            print(torch.max(next_token_probs))
            next_token = torch.argmax(next_token_probs, dim=-1).item()
            next_token_tensor = torch.tensor(
                [[next_token]], device=inputs.device
            )
            inputs = torch.cat((inputs, next_token_tensor), dim=1)
            print(next_token, model.tokenizer.batch_decode(inputs))
            eos_position = get_eos_position(next_token_probs, eos_token_id)
            eos_poss.append(eos_position)
    torch.cuda.empty_cache()
    return eos_poss


def get_eos_position(mode_result: torch.Tensor, eos_token_id: int = 2):
    eos_probability = mode_result[:, eos_token_id]
    sort_result = torch.sort(mode_result, descending=True)
    eos_position = torch.where(sort_result[0] == eos_probability)[1][0]
    return eos_position


def test_max_model_output_length(large_model: Model, prompt: List[str]):
    seq = prompt[0]
    inputs = large_model.tokenizer(seq, return_tensors="pt")
    inputs.to("cuda:1")
    model_eos = 2
    eos_poss = predict(large_model, inputs.input_ids, model_eos)

    return eos_poss


def get_prompt():
    dataset_path = (
        "/root/vllm/dataset/ShareGPT_V3_unfiltered_cleaned_split.json"
    )
    selected_seqs = []

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
        random.seed(2)
        random.shuffle(dataset)
        for i in range(len(dataset)):
            if len(set(selected_seqs)) == 10:
                break

            # Tokenize the prompts and completions.
            prompt = dataset[i][0]
            selected_seqs.append(prompt)
        return selected_seqs


if __name__ == "__main__":
    test_prompts = get_prompt()
    # large_model_name = "JackFram/llama-160m"
    large_model_name = "meta-llama/Llama-2-13b-hf"
    large_model = load_model(large_model_name)
    # eos_probability(large_model, small_model, test_prompts)
    eos_poss = test_max_model_output_length(large_model, test_prompts)
    print(eos_poss)
