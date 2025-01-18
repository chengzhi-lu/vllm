# import torch

# # Create a multidimensional tensor (e.g., a MxNxL tensor)

# M, N, P = 300, 200, 500
# tensor1 = torch.randn(M, N, P)
# tensor1.to('cuda')
# tensor2 = torch.randn(P, N)
# tensor2.to('cuda')

# org_times = []
# for i in range(0, M):
#     # record the cost of tensorxtensor
#     torch.cuda.Event(enable_timing=True)
#     start = torch.cuda.Event(enable_timing=True)
#     end = torch.cuda.Event(enable_timing=True)
#     start.record()
#     result = torch.matmul(tensor1, tensor2)
#     end.record()
#     torch.cuda.synchronize()
#     org_time = start.elapsed_time(end)
#     org_times.append(org_time)

# modified_times = []
# # Set some values to zero, e.g. N values in the first row
# tensor1[:10, :, :] = 0  # 将前10行设置为全0

# for i in range(0, M):
#     # record the cost of tensorxtensor
#     torch.cuda.Event(enable_timing=True)
#     start = torch.cuda.Event(enable_timing=True)
#     end = torch.cuda.Event(enable_timing=True)
#     start.record()
#     result = torch.matmul(tensor1, tensor2)
#     end.record()
#     torch.cuda.synchronize()
#     modified_time = start.elapsed_time(end)
#     modified_times.append(modified_time)

# print("Mean original time: ", sum(org_times) / len(org_times))
# print("Mean modified time: ", sum(modified_times) / len(modified_times))


import torch, time
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
# 设置随机数
torch.manual_seed(0) 
# np.random.seed(0)

# 设置参数 for Llama2-13B on one A100 GPU
batch_size = 8
sequence_length = 350
head_size = 64
num_heads = 40
embed_size_per_head = 128
num_layers = 40
min_cache_layers = 10
max_cache_layers = 25
recom_num_layers = num_layers - min_cache_layers

loops = 100
org_times = []

# 初始化Query, Key, Value
Q = torch.randn(batch_size, sequence_length, head_size, num_heads).to('cuda')  
K = torch.randn(batch_size, sequence_length, head_size, num_heads).to('cuda')
V = torch.randn(batch_size, sequence_length, head_size, num_heads).to('cuda')


mask_tensor = torch.ones_like(Q)  # 初始化为全 1
mask_tensor[:batch_size-1] = 0    # 将前 bs-1 行设置为 0

Q_m = Q*mask_tensor
K_m = K*mask_tensor
V_m = V*mask_tensor
for i in range(loops):
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    start_time.record()
    d_k = embed_size_per_head
    scores = torch.matmul(Q_m, K_m.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    weights = F.softmax(scores, dim=-1)
    output = torch.matmul(weights, V_m)
    end_time.record()
    torch.cuda.synchronize()
    org_times.append(start_time.elapsed_time(end_time))

time.sleep(10)

modified_times = []

for i in range(loops):
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    start_time.record()
    d_k = embed_size_per_head
    Q_s = Q[-1:]
    K_s = K[-1:]
    V_s = V[-1:]
    scores = torch.matmul(Q_s, K_s.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    weights = F.softmax(scores, dim=-1)
    output = torch.matmul(weights, V_s)
    end_time.record()
    torch.cuda.synchronize()
    modified_times.append(start_time.elapsed_time(end_time))

print("Mean original time: ", sum(org_times) / len(org_times), "ms")
print("Mean modified time: ", sum(modified_times) / len(modified_times), "ms")
time.sleep(10)

# save to csv
mean_org_time = sum(org_times) / len(org_times)
mean_modified_time = sum(modified_times) / len(modified_times)
data = {
    'sequence_length': [sequence_length],
    'max_cache_layers': [max_cache_layers],
    'min_cache_layers': [min_cache_layers],
    'mean_org_time': [mean_org_time],
    'mean_modified_time': [mean_modified_time],
    'delta': [mean_org_time - mean_modified_time],
    'speedup': [(mean_org_time - mean_modified_time)/mean_org_time]
}
df = pd.DataFrame(data)
df.to_csv('test_policy.csv', mode='a', header=not os.path.exists('test_policy.csv'), index=False)