# use JackFram/llama-68m in the huggingface to predict the eos of the prompt
from transformers import AutoTokenizer, LlamaForCausalLM
from typing import List
import json
import random
from dataclasses import dataclass
import tqdm

@dataclass
class Model:
    model: LlamaForCausalLM
    tokenizer: AutoTokenizer


def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    m = LlamaForCausalLM.from_pretrained(model_name, device_map="cuda:1")
    model = Model(m, tokenizer)
    return model


def predict(model: Model, prompt):
    inputs = model.tokenizer(prompt, return_tensors="pt")
    inputs.to("cuda:1")
    generate_ids = model.model.generate(
        inputs.input_ids, max_length=len(inputs.input_ids[0]) + 1
    )
    result = model.tokenizer.batch_decode(
        generate_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    if model.tokenizer(result).input_ids[-1] == model.tokenizer.eos_token_id:
        return "++++++"
    return result


def compare_result(small_model: Model, large_model: Model, prompt: List[str]):
    false_positive_count = 0  # small_model is not eos and large_model is eos
    false_negtive_count = 0  # small_model is  eos and large_model is not eos
    total_count = 0
    for i in tqdm(range(len(prompt))):
        # compare the output of the two model and loop until the output of large_model is eos
        seq = prompt[i]
        result1 = predict(small_model, seq)
        result2 = predict(large_model, seq)
        while result2 != "++++++":
            if result1 == "++++++" and result2 != "++++++":
                false_positive_count += 1

            seq = seq + result2
            result1 = predict(small_model, seq)
            result2 = predict(large_model, seq)
            total_count += 1
        if result1 != "++++++" and result2 == "++++++":
            false_negtive_count += 1

    return false_positive_count, false_negtive_count, total_count


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
            if len(set(selected_seqs)) == 100:
                break

            # Tokenize the prompts and completions.
            prompt = dataset[i][0]
            selected_seqs.append(prompt)
        return selected_seqs


if __name__ == "__main__":
    test_prompts = get_prompt()
    small_model = "JackFram/llama-68m"
    large_model = "meta-llama/Llama-2-13b-hf"
    small_model = load_model(small_model)
    large_model = load_model(small_model)
    print(test_prompts)
    false_positive_count, false_negtive_count, total_count = compare_result(
        small_model, large_model, test_prompts
    )
    print(false_negtive_count, false_positive_count, total_count)
