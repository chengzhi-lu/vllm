import json
import os

file_path_1 = "vllm-2.0qps-Llama-2-13b-chat-hf-203550-fcfs.json"
file_path_2 = "vllm-2.0qps-Llama-2-13b-chat-hf-201350-tfittradeoff.json"


with open(file_path_1, 'r', encoding="utf-8") as file1:
    data1 = json.load(file1)
# print(type(data))
input_lens_list1 = data1["input_lens"]
output_lens_list1 = data1["output_lens"]

with open(file_path_2, 'r', encoding="utf-8") as file2:
    data2 = json.load(file2)
# print(type(data))
input_lens_list2 = data1["input_lens"]
output_lens_list2 = data1["output_lens"]

print(input_lens_list1 == input_lens_list2)

