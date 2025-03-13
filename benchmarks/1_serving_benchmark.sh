#!/bin/bash
set -euo pipefail

# --------------------------
# 配置参数
# --------------------------
# 基础配置
COUNTER_FILE=".counter.txt"
result_dir="$(pwd)/result"
TOKENIZERS_PARALLELISM="true"

# 模型和数据集配置
model_names=(
  "meta-llama/Llama-2-13b-chat-hf"
  # "meta-llama/Llama-2-70b-chat-hf"
)
parallel_types=(
  "single"
  # "tp"
  # "pp"
)
datasets=(
  "sharegpt /root/vllm/dataset/ShareGPT_V3_unfiltered_cleaned_split.json"
  # "alpaca /root/vllm/dataset/alpaca_data.json"
)

# 服务器配置
swap_space=20
preemption_mode="swap"
gpu_memory_utilization=0.6
max_tokens=4096
max_num_seqs=256
max_serving_time=86400
num_shared_blocks=0

# 测试策略组合
scheduler_swap_policies=(
  "tfittradeoff partial"
  # "fcfs full"
  # "sjf full" 
  # "sjmlfq full"
  "las full"
)

request_rates=(1 2 4 8 16 32)
swap_out_partial_rates=(0.5)

# --------------------------
# 初始化计数器
# --------------------------
if [[ -f "$COUNTER_FILE" ]]; then
  COUNTER=$(cat "$COUNTER_FILE")
else
  COUNTER=0
fi
COUNTER=$((COUNTER + 1))
echo "$COUNTER" > "$COUNTER_FILE"

# --------------------------
# 功能函数定义
# --------------------------
start_server() {
  local policy=$1
  local swap_policy=$2
  local swap_out_partial_rate=$3
  local parallel_type=$4
  local model_name=$5

  local log_file="api_server_${model_name##*/}_${parallel_type}_${policy}_${swap_policy}.log"
  
  echo "启动服务器: model=${model_name##*/} type=$parallel_type policy=$policy"
  local parallel_args=""
  # GPU分配逻辑
    if [[ "$parallel_type" == @("pp"|"tp") ]]; then
        gpu_devices="0,1,2,3"
    elif [[ "$parallel_type" == "single" && "$model_name" == "meta-llama/Llama-2-13b-chat-hf" ]]; then
        gpu_devices="0"
    else
        continue
    fi
  IFS=',' read -ra gpu_array <<< "$gpu_devices"
  num_gpus=${#gpu_array[@]}

  case "$parallel_type" in
    "single")
      parallel_args="--tensor-parallel-size $num_gpus"
      ;;
    "pp")
      parallel_args="--pipeline-parallel-size $num_gpus --worker-use-ray"
      ;;
    "tp")
      parallel_args="--tensor-parallel-size $num_gpus --worker-use-ray"
      ;;
  esac

  CUDA_VISIBLE_DEVICES=$gpu_devices RAY_DEDUP_LOGS=0 taskset -c 28-29 python3 -m vllm.entrypoints.openai.api_server \
    --model "$model_name" \
    $parallel_args \
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
  server_pid=$!

  sleep 20  # 等待服务器初始化
}

run_benchmark() {
  local policy=$1
  local swap_policy=$2
  local swap_out_partial_rate=$3
  local request_rate=$4
  local dataset_path=$5
  local dataset_name=$6
  local model_name=$7
  local parallel_type=$8
  local total_request_nums=1024

  local request_duration=$((total_request_nums / request_rate))
  local log_file="benchmark_${model_name##*/}_${parallel_type}_${policy}_rate${request_rate}.log"

  echo "运行基准测试: model=${model_name##*/} rate=$request_rate"
  taskset -c 30-49 python3 benchmark_serving.py \
    --execution-counter "$COUNTER" \
    --dataset-path "$dataset_path" \
    --dataset-name "$dataset_name" \
    --request-rate "$request_rate" \
    --num-prompts 3000 \
    --request-duration "$request_duration" \
    --sharegpt-output-len 2000 \
    --model "$model_name" \
    --scheduler-policy "$policy" \
    --save-result \
    --result-dir "$result_dir" \
    --metadata "model=${model_name##*/}" \
               "parallel_type=$parallel_type" \
               "swap_space=$swap_space" \
               "preemption_mode=$preemption_mode" \
               "scheduler_policy=$policy" \
               "gpu_memory_utilization=$gpu_memory_utilization" \
               "max_num_seqs=$max_num_seqs" \
               "max_tokens=$max_tokens" \
               "swap_policy=$swap_policy" \
               "swap_out_partial_rate=$swap_out_partial_rate"  >> "$log_file" 2>&1

  python3 parse_log.py \
    --policy "$policy" \
    --swap-policy "$swap_policy" \
    --result-dir "$result_dir" \
    --execution-counter "$COUNTER" \
    --request-rate "$request_rate" \
    --swap-out-partial-rate "$swap_out_partial_rate" \
    --model "$model_name" \
    --parallel-type "$parallel_type"
}

# --------------------------
# 主执行流程
# --------------------------

for ptype in "${parallel_types[@]}"; do
    for swap_out_partial_rate in "${swap_out_partial_rates[@]}"; do
      for scheduler_swap_policy in "${scheduler_swap_policies[@]}"; do
        IFS=' ' read -r policy swap_policy <<< "$scheduler_swap_policy"
        
        for dataset in "${datasets[@]}"; do
          IFS=' ' read -r dataset_name dataset_path <<< "$dataset"
          
          for model_name in "${model_names[@]}"; do
            # 启动服务
            start_server "$policy" "$swap_policy" "$swap_out_partial_rate" \
            "$ptype" "$model_name"
            
            # 运行基准测试
            for request_rate in "${request_rates[@]}"; do
              run_benchmark "$policy" "$swap_policy" "$swap_out_partial_rate" \
                 "$request_rate" "$dataset_path" "$dataset_name" \
                "$model_name" "$ptype"
              sleep 15
            done
            
            # 停止服务器
            echo "停止服务器: PID=$server_pid"
            kill "$server_pid" && wait "$server_pid" 2>/dev/null
            sleep 10
        done
      done
    done
  done
done

echo "所有测试完成，结果保存在: $result_dir"