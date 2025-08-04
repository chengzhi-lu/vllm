import pandas as pd
import json

import multiprocessing
from multiprocessing import Process, Queue
from transformers import AutoTokenizer


def producer(queue, dataset_path, data_type):
    import random

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    key_name = "conversation" if data_type == "Alpaca" else "conversations"
    dataset = [data for data in dataset if len(data[key_name]) >= 2]
    random.seed(42)
    sampled_dataset = random.sample(dataset, 15000) if len(dataset) > 15000 else dataset
    for data in sampled_dataset:
        conversations = data.get(key_name, [])
        if len(conversations) >= 2:
            for i in range(len(conversations) - 1):
                if conversations[i]["from"] == "human" and conversations[i + 1]["from"] == "gpt":
                    prompt = conversations[i]["value"]
                    completion = conversations[i + 1]["value"]
                    queue.put({"prompt": prompt, "completion": completion})
    # 在队列中放入None作为结束标记
    for _ in range(multiprocessing.cpu_count()):
        queue.put(None)


def consumer(queue, tokenizer_id, model_id, trust_remote_code, output_queue):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=trust_remote_code)
    while True:
        item = queue.get()
        if item is None:  # 结束信号
            output_queue.put(None)
            break

        conversation = item
        input_data = conversation["prompt"]
        output_data = conversation["completion"]
        length_types = []
        data_lengths = []
        model_ids = []
        data_types = []
        input_length = len(tokenizer(input_data).input_ids)
        output_length = len(tokenizer(output_data).input_ids)
        if input_length <= 4096 and output_length <= 2048:
            input_length = len(input_data)
            output_length = len(output_data)
            length_types.append("input")
            data_lengths.append(input_length)
            model_ids.append(model_id)
            data_types.append(data_type)
            length_types.append("output")
            data_lengths.append(output_length)
            model_ids.append(model_id)
            data_types.append(data_type)

        output_queue.put((length_types, data_lengths, model_ids, data_types))


def main(dataset_path, tokenizer_id, data_type, model_id, trust_remote_code):
    input_queue = Queue(maxsize=100)
    output_queue = Queue()

    # 启动生产者
    p_producer = Process(target=producer, args=(input_queue, dataset_path, data_type))
    p_producer.start()

    # 启动消费者
    consumers = []
    consumer_numbers = min(multiprocessing.cpu_count(), 50)
    for i in range(consumer_numbers):
        p = Process(target=consumer, args=(input_queue, tokenizer_id, model_id, trust_remote_code, output_queue))
        p.start()
        consumers.append(p)

    # 收集结果
    all_results = []
    completed_consumers = 0
    while True:
        result = output_queue.get()
        if result is None:
            completed_consumers += 1
            if completed_consumers == consumer_numbers:
                break
        else:
            length_types, data_lengths, model_ids, data_types = result
            all_results.extend(list(zip(length_types, data_lengths, model_ids, data_types)))

    # 等待所有进程完成
    p_producer.join()
    for p in consumers:
        p.join()

    import pandas as pd

    df = pd.DataFrame(all_results, columns=["length_type", "data_length", "model_id", "data_type"])

    return df


# 示例调用


if __name__ == "__main__":
    models = ["meta-llama/Llama-2-13b-hf", "meta-llama/Llama-2-70b-hf"]
    dataset_path = [
        "/root/vllm/dataset/ShareGPT_V3_unfiltered_cleaned_split.json",
        # "/root/vllm/dataset/paper_assistant_transformed.json",
        "/root/vllm/dataset/lmsys-chat-1m-aligned.json",
    ]
    data_types = ["ShareGPT", "LMSys-Chat"]
    df = None
    for model_id in models:
        for i, data_type in enumerate(data_types):
            # tmp_df = get_all_data(model_id, dataset_path[i], data_type)
            tmp_df = main(
                dataset_path=dataset_path[i],
                tokenizer_id=model_id,
                data_type=data_type,  # 或其它类型
                model_id=model_id,
                trust_remote_code=False,
            )
            if df is None:
                df = tmp_df
            else:
                df = pd.concat([df, tmp_df], ignore_index=True)

    # print(df)
    df.to_csv(
        "/root/vllm/benchmarks/result/fixed_result/dataset_distribution/dataset_distribution_updated.csv", index=False
    )
