#!/bin/bash
set -euo pipefail

# --------------------------
# 配置参数
# --------------------------
result_dir="$(pwd)/result"
TOKENIZERS_PARALLELISM="true"

# 远程传入参数模型和数据集配置
model_name=$1
max_num_seqs=$2
# 测试策略组合
policy=$3
swap_policy=$4
swap_out_partial_rate=$5


# 默认服务器配置
swap_space=20
preemption_mode="swap"
gpu_memory_utilization=0.9
max_tokens=16384
max_serving_time=86400
num_shared_blocks=0
swap_out_partial_rates=(0.5)
gpu_devices="0,1,2,3"

# --------------------------
# 功能函数定义
# --------------------------
start_server() {
  local policy=$1
  local swap_policy=$2
  local swap_out_partial_rate=$3
  local parallel_type=$4
  local model_name=$5

  local log_file="/root/vllm/benchmarks/logs/api_server_${model_name##*/}_${parallel_type}_${policy}_${swap_policy}.log"
  
  local parallel_args=""
  
  echo "启动服务器: model=${model_name##*/} type=$parallel_type policy=$policy batch size=$max_num_seqs"

  parallel_args="--pipeline-parallel-size 8 --worker-use-ray"


  CUDA_VISIBLE_DEVICES=$gpu_devices RAY_DEDUP_LOGS=0 taskset -c 28-29 /usr/bin/python3 -m vllm.entrypoints.openai.api_server \
    --model "$model_name" \
    $parallel_args \
    --parallel-type $parallel_type\
    --swap-space "$swap_space" \
    --preemption-mode "$preemption_mode" \
    --scheduler-policy "$policy" \
    --enforce-eager \
    --enable-chunked-prefill \
    --max-num-batched-tokens "$max_tokens" \
    --max-num-seqs "$max_num_seqs" \
    --swap-out-tokens-policy "$swap_policy" \
    --swap-out-partial-rate "$swap_out_partial_rate" \
    --num-shared-blocks "$num_shared_blocks" \
    --gpu-memory-utilization "$gpu_memory_utilization" \
    --disable-sliding-window \
    --disable-log-requests \
    --max-serving-time "$max_serving_time" > "$log_file" 2>&1 &
}


# --------------------------
# 主执行流程
# --------------------------
            
# 启动服务
start_server "$policy" "$swap_policy" "$swap_out_partial_rate" "pp" "$model_name"


