import json
from datasets import load_dataset
import multiprocessing
from multiprocessing import Queue, Process
from tqdm import tqdm
from rich.progress import track
import rich.progress


# Login using e.g. `huggingface-cli login` to access this dataset
def download_dataset():
    dataset_dict = load_dataset("lmsys/lmsys-chat-1m")
    converted = [ds.to_list() for split, ds in dataset_dict.items()]
    with open("lmsys-chat-1m.json", "w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=4)


def align_dataset():
    with rich.progress.open("lmsys-chat-1m.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)
    conversations = []
    for data in track(dataset[0], description="Processing..."):
        conversation = {}
        conversation["id"] = data["conversation_id"]
        conversation["conversations"] = []
        for conv in data["conversation"]:
            if conv["role"] == "user":
                conversation["conversations"].append({"from": "human", "value": conv["content"]})
            else:
                conversation["conversations"].append({"from": "gpt", "value": conv["content"]})
        conversations.append(conversation)
    with open("lmsys-chat-1m-aligned.json", "w", encoding="utf-8") as f:
        json.dump(conversations, f, ensure_ascii=False, indent=4)
    print("处理完成！共处理对话记录:", len(conversations))


def align_dataset_producer_consumer():
    # 定义队列大小
    QUEUE_SIZE = 100
    # 读取原始数据
    with open("lmsys-chat-1m.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # 展平所有数据（假设只处理第一个 split）
    raw_data_list = dataset[0]

    # 创建队列
    data_queue = Queue(maxsize=QUEUE_SIZE)
    results_queue = Queue()

    # 生产者函数
    def producer(data_list, data_queue):
        for data in data_list:
            data_queue.put(data)  # 将数据放入队列
        # 发送结束信号（发送多个 None 表示多个消费者可以退出）
        for _ in range(NUM_CONSUMER_PROCESSES):
            data_queue.put(None)

    # 消费者函数
    def consumer(data_queue, results_queue):
        while True:
            data = data_queue.get()
            if data is None:  # 结束信号
                break
            # 处理单条数据
            conversation = {"id": data["conversation_id"], "conversations": []}
            for conv in data["conversation"]:
                role = "human" if conv["role"] == "user" else "gpt"
                conversation["conversations"].append({"from": role, "value": conv["content"]})
            # 将结果加入结果队列
            results_queue.put(conversation)

    # 启动生产者和消费者进程
    NUM_CONSUMER_PROCESSES = multiprocessing.cpu_count() - 1  # 根据 CPU 核心数调整

    producer_process = Process(target=producer, args=(raw_data_list, data_queue))
    consumer_processes = [
        Process(target=consumer, args=(data_queue, results_queue)) for _ in range(NUM_CONSUMER_PROCESSES)
    ]

    producer_process.start()
    for p in consumer_processes:
        p.start()

    # 等待所有消费者完成工作
    for p in consumer_processes:
        p.join()

    # 收集所有的结果
    results = []
    while not results_queue.empty():
        results.append(results_queue.get())

    # 写入最终文件
    with open("lmsys-chat-1m-aligned.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print("处理完成！共处理对话记录:", len(results))


if __name__ == "__main__":
    # align_dataset_producer_consumer()
    align_dataset()
