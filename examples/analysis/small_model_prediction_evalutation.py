# use JackFram/llama-68m in the huggingface to predict the eos of the prompt
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
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
    m = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda:1")
    model = Model(model_name, m, tokenizer)
    return model


def predict(model: Model, inputs: torch.Tensor):
    input_length = len(inputs[0])

    generate_ids = model.model.generate(
        inputs,
        max_length=len(inputs[0]) + 4,
        repetition_penalty=1.2,
    )
    generate_ids = generate_ids[:, : input_length + 1]
    torch.cuda.empty_cache()
    return generate_ids


def compare_result(small_model: Model, large_model: Model, prompt: List[str]):
    false_positive_count = 0  # small_model is not eos and large_model is eos
    false_negtive_count = 0  # small_model is  eos and large_model is not eos
    true_positive_count = 0
    total_count = 0
    for i in tqdm(range(len(prompt))):
        # compare the output of the two model and loop until the output of large_model is eos
        seq = prompt[i]
        inputs = large_model.tokenizer(seq, return_tensors="pt")
        inputs.to("cuda:1")
        small_model_generated_ids = predict(small_model, inputs.input_ids)
        large_model_generated_ids = predict(large_model, inputs.input_ids)
        large_model_result = large_model_generated_ids[0][-1]
        small_model_result = small_model_generated_ids[0][-1]
        large_model_eos = 2
        small_model_eos = 2
        while (
            large_model_result != large_model_eos
            and len(large_model_generated_ids[0]) < 500
        ):
            if (
                small_model_result == small_model_eos
                and large_model_result != large_model_eos
            ):
                false_positive_count += 1
            elif (
                small_model_result == small_model_eos
                and large_model_result == large_model_eos
            ):
                true_positive_count += 1
            print(f"\n+++++++\n{large_model_generated_ids},\n++++++++++\n")
            small_model_generated_ids = predict(
                small_model, large_model_generated_ids
            )
            large_model_generated_ids = predict(
                large_model, large_model_generated_ids
            )
            small_model_result = small_model_generated_ids[0][-1]
            large_model_result = large_model_generated_ids[0][-1]
            print(
                f"\n========\n{small_model_generated_ids},\n----------\n{large_model_generated_ids},\n========\n"
            )
            total_count += 1
        print("result1", small_model_result, "result2", large_model_result)
        if (
            small_model_result != small_model_eos
            and large_model_result == large_model_eos
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
            if len(set(selected_seqs)) == 1:
                break

            # Tokenize the prompts and completions.
            prompt = dataset[i][0]
            selected_seqs.append(prompt)
        return selected_seqs


if __name__ == "__main__":
    test_prompts = get_prompt()
    small_model_name = "JackFram/llama-68m"
    large_model_name = "meta-llama/Llama-2-13b-hf"
    small_model = load_model(small_model_name)
    large_model = load_model(large_model_name)
    (
        false_positive_count,
        false_negtive_count,
        true_positive,
        total_count,
    ) = compare_result(small_model, large_model, test_prompts)
    print(
        f"false_positive_count: {false_positive_count}, false_negtive_count: {false_negtive_count}, true_positive: {true_positive}, total_count: {total_count}"
    )
