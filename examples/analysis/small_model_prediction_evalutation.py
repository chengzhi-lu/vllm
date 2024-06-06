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


# def load_all_model(model_name):
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     agent_model_config = model_config = AutoConfig.from_pretrained(model_name)
#     model = AutoModelForCausalLM.from_config(model_config)
#     agent_model_config.num_hidden_layers = 1
#     agent_model = AutoModelForCausalLM.from_config(model_config)
#     model.to("cuda:1")
#     agent_model.to("cuda:1")
#     large_model = Model(model_name, model, tokenizer)
#     small_model = Model(model_name, agent_model, tokenizer)
#     return large_model, small_model


def predict(model: Model, inputs: torch.Tensor, eos_token_id: int = 2):
    print(inputs.shape)
    with torch.no_grad():
        generate_ids = model.model(
            inputs,
            # use_cache=True,
            # max_length=len(inputs[0]) + max_output_length,
            # pad_token_id=model.tokenizer.pad_token_id
            # if model.tokenizer.pad_token_id is not None
            # else 0,
            # repetition_penalty=1.2,
            return_dict=True,
        )
    logits = generate_ids["logits"]
    probs = torch.nn.functional.softmax(logits, dim=-1)
    next_token_probs = probs[:, -1, eos_token_id]
    # eos_prob = next_token_probs[eos_token_id]
    torch.cuda.empty_cache()
    return next_token_probs


def eos_probability(large_model: Model, small_model: Model, prompt: List[str]):
    inputs = large_model.tokenizer(prompt,padding=True, return_tensors="pt")
    inputs.to("cuda:1")
    with torch.no_grad():
        print(predict(small_model, inputs.input_ids, 2))


def test_max_model_output_length(large_model: Model, prompt: List[str]):
    seq = prompt[0]
    inputs = large_model.tokenizer(seq, return_tensors="pt")
    input_length = len(inputs.input_ids[0])
    inputs.to("cuda:1")
    for seq in prompt:
        model_result = predict(large_model, inputs.input_ids, 10000)
    model_result = predict(large_model, inputs.input_ids, 1)
    model_result_last_token = model_result[0][input_length]
    input_length = input_length + 1
    model_result = model_result[:, :input_length]
    model_eos = 2
    while model_result_last_token != model_eos:
        model_result = predict(large_model, model_result, 1)
        model_result_last_token = model_result[0][input_length]
        input_length = input_length + 1
        model_result = model_result[:, :input_length]


def compare_result(small_model: Model, large_model: Model, prompt: List[str]):
    false_positive_count = 0  # small_model is not eos and large_model is eos
    false_negtive_count = 0  # small_model is  eos and large_model is not eos
    true_positive_count = 0
    total_count = 0
    for i in tqdm(range(len(prompt))):
        # compare the output of the two model and loop until the output of large_model is eos
        seq = prompt[i]
        inputs = large_model.tokenizer(seq, return_tensors="pt")
        input_length = len(inputs.input_ids[0])
        inputs.to("cuda:1")
        small_model_generated_ids = predict(small_model, inputs.input_ids)
        large_model_generated_ids = predict(large_model, inputs.input_ids)
        large_model_result_last_token = large_model_generated_ids[0][
            input_length
        ]
        small_model_result_last_token = small_model_generated_ids[0][
            input_length
        ]
        input_length = input_length + 1
        large_model_generated_ids = large_model_generated_ids[:, :input_length]

        large_model_eos = small_model_eos = 2
        while (
            int(large_model_result_last_token) != large_model_eos
            and input_length < 500
        ):
            if (
                small_model_result_last_token == small_model_eos
                and large_model_result_last_token != large_model_eos
            ):
                false_positive_count += 1
            elif (
                small_model_result_last_token == small_model_eos
                and large_model_result_last_token == large_model_eos
            ):
                true_positive_count += 1
            small_model_generated_ids = predict(
                small_model, large_model_generated_ids, 1
            )
            large_model_generated_ids = predict(
                large_model, large_model_generated_ids, 1
            )
            small_model_result_last_token = small_model_generated_ids[0][
                input_length
            ]
            large_model_result_last_token = large_model_generated_ids[0][
                input_length
            ]
            print(
                "large_model_result_last_token",
                large_model_result_last_token,
                "small_model_result_last_token",
                small_model_result_last_token,
            )
            input_length = input_length + 1
            large_model_generated_ids = large_model_generated_ids[
                :, :input_length
            ]

            total_count += 1
        print(
            "result1",
            small_model_result_last_token,
            "result2",
            large_model_result_last_token,
        )
        if (
            small_model_result_last_token != small_model_eos
            and large_model_result_last_token == large_model_eos
        ):
            false_negtive_count += 1
        if len(seq) >= 500:
            print("seq is too long")
    return (
        false_positive_count,
        false_negtive_count,
        true_positive_count,
        total_count,
    )


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
        random.seed(1)
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
    small_model_name = "JackFram/llama-160m"
    large_model_name = "JackFram/llama-160m"
    # large_model_name = "meta-llama/Llama-2-13b-hf"
    # large_model, small_model = load_all_model(large_model_name)
    # print(large_model)
    # print(small_model)
    small_model = load_model(small_model_name)
    large_model = load_model(large_model_name)
    eos_probability(large_model, small_model, test_prompts)
    # test_max_model_output_length(large_model, test_prompts)
    # (
    #     false_positive_count,
    #     false_negtive_count,
    #     true_positive,
    #     total_count,
    # ) = compare_result(small_model, large_model, test_prompts)
    # print(
    #     f"false_positive_count: {false_positive_count}, false_negtive_count: {false_negtive_count}, true_positive: {true_positive}, total_count: {total_count}"
    # )
