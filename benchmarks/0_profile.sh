#!/bin/bash
set -euo pipefail
source $(pwd)/shared_functions.sh
# --------------------------
# 配置参数
# --------------------------
# 基础配置
COUNTER=9
result_dir="$(pwd)/result"
TOKENIZERS_PARALLELISM="true"

# 模型和数据集配置
model_names=(
    # "meta-llama/Llama-2-70b-chat-hf"
    "meta-llama/Llama-2-13b-chat-hf"
)
parallel_types=("pp")
datasets=(
    "sharegpt /root/vllm/dataset/ShareGPT_V3_unfiltered_cleaned_split.json"
    # "alpaca /root/vllm/dataset/alpaca_data.json"
)

# 服务器配置
swap_space=20
preemption_mode="swap"
gpu_memory_utilization=0.6
max_tokens=4096
max_num_seqs=512
max_serving_time=86400
num_shared_blocks=0
max_request_nums=4096
# 测试策略组合
scheduler_swap_policies=(
    "fcfs full"
)

request_rates=(20)
swap_out_partial_rates=(0.5)

host="10.119.46.54"

# --------------------------
# 功能函数定义
# --------------------------

parse_result() {
  local policy=$1
  local swap_policy=$2
  local request_rate=$3
  local model_name=$4
  local parallel_type=$5

  python3 parse_profile_log.py \
    --policy "$policy" \
    --swap-policy "$swap_policy" \
    --result-dir "$result_dir" \
    --execution-counter "$COUNTER" \
    --request-rate "$request_rate" \
    --model "$model_name" \
    --parallel-type "$parallel_type" \
    --num-instances 8
}

copy_result() {
  scp -r /root/vllm/vllm/core/profile_data lucz@10.119.46.53:/home/lucz/github/
  ssh lucz@10.119.46.53 'docker cp /home/lucz/github/profile_data vllm_lucz:/root/vllm/vllm/core/'
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
          # 跳过70b模型的single并行类型
          if [[ "$model_name" == "meta-llama/Llama-2-70b-chat-hf" && "$ptype" == "single" ]]; then
            echo "跳过 llama2-70b 的 single 并行类型测试"
            continue
          fi
          
          # 启动服务
          start_server "$policy" "$swap_policy" "$swap_out_partial_rate" \
                       "$ptype" "$model_name"
          
          # 运行基准测试
          for request_rate in "${request_rates[@]}"; do
            run_benchmark "$policy" "$swap_policy" "$swap_out_partial_rate" \
                          "$request_rate" "$dataset_path" "$dataset_name" \
                          "$model_name" "$ptype" "$max_request_nums"
            sleep 5
            parse_result "$policy" "$swap_policy" "$request_rate" \
                          "$model_name" "$ptype" 
            copy_result
          done
          
          # 停止服务器
          echo "停止服务器"
          terminate_server "$ptype"

          sleep 5
        done
      done
    done
  done
done

echo "所有测试完成，结果保存在: $result_dir"