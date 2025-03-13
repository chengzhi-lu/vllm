# 读取当前计数器的值
COUNTER=9

# start vllm server
pwd=`pwd`

dataset_name="sharegpt"
dataset_path="/root/vllm/dataset/ShareGPT_V3_unfiltered_cleaned_split.json"
# dataset_name='alpaca'
# dataset_path="/root/vllm/dataset/alpaca_data.json"
result_dir="${pwd}/result"


# 模型和参数配置
declare -a models=(
    "meta-llama/Llama-2-70b-chat-hf"
    # "meta-llama/Llama-2-13b-chat-hf"
)

declare -a scheduler_swap_policies=(
    "fcfs full"
)

declare -a parallel_types=("pp")



preemption_mode="swap"
gpu_memory_utilization=0.6
max_num_seqs=512
swap_space=20
max_tokens=4096
max_serving_time=86400 # 86400
total_request_nums=1024
TOKENIZERS_PARALLELISM=true
request_rates[0]=20

swap_out_partial_rates=(0.5)



start_server() {
    local parallel_args=$1
    model=$(echo "$model_name" | awk -F'/' '{print $2}')
    CUDA_VISIBLE_DEVICES=$gpu_devices RAY_DEDUP_LOGS=0 taskset -c 28-29 python3 -m vllm.entrypoints.openai.api_server \
        --model $model_name --swap-space $swap_space --preemption-mode $preemption_mode \
        --scheduler-policy $policy --enforce-eager --max-num-batched-tokens $max_tokens \
        --max-num-seqs $max_num_seqs --swap-out-tokens-policy $swap_policy \
        --swap-out-partial-rate $swap_out_partial_rate $parallel_args \
        --gpu-memory-utilization $gpu_memory_utilization \
        --disable-sliding-window --disable-log-requests --max-serving-time $max_serving_time \
        > "result/profile/api_server_${policy}_${swap_policy}_${model}_${parallel_type}_profile.log" 2>&1 &
    pid=$!
}



for model_name in "${models[@]}"; do
    for parallel_type in "${parallel_types[@]}"; do
        # GPU分配逻辑
        if [[ "$parallel_type" == @("pp"|"tp") ]]; then
            gpu_devices="0,1,2,3"
        elif [[ "$parallel_type" == "single" && "$model_name" == "meta-llama/Llama-2-13b-chat-hf" ]]; then
            gpu_devices="0"
        else
            continue
        fi

        num_gpus=$(tr -cd ',' <<< "$gpu_devices" | wc -c)
        ((num_gpus++))

        for swap_out_partial_rate in "${swap_out_partial_rates[@]}"; do
            for scheduler_swap_policy in "${scheduler_swap_policies[@]}"; do
                IFS=' ' read -r policy swap_policy <<< "$scheduler_swap_policy"
                
                # # 启动服务器
                case "$parallel_type" in
                    "single")
                        start_server "--tensor-parallel-size $num_gpus"
                        ;;
                    "pp")
                        start_server "--worker-use-ray --pipeline-parallel-size $num_gpus"
                        ;;
                    "tp")
                        start_server "--tensor-parallel-size $num_gpus --worker-use-ray"
                        ;;
                esac

                # 压测逻辑
                for request_rate in "${request_rates[@]}"; do
                    request_duration=$((total_request_nums / request_rate))
                    for i in {0..0}; do
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
                            --metadata "swap_space=$swap_space preemption_mode=$preemption_mode scheduler_policy=$policy gpu_memory_utilization=$gpu_memory_utilization max_num_seqs=$max_num_seqs max_tokens=$max_tokens swap_policy=$swap_policy iter_threshold=$iter_threshold swap_out_partial_rate=$swap_out_partial_rate parallel_type=$parallel_type" \
                            >> "benchmark-${policy}.log" 2>&1
                        
                        sleep 3
                        echo "Start to parse ${policy} ${swap_policy}"
                        python3 parse_profile_log.py \
                            --policy "$policy" \
                            --swap-policy "$swap_policy" \
                            --result-dir "$result_dir" \
                            --execution-counter "$COUNTER" \
                            --request-rate "$request_rate" \
                            --model "$model_name" \
                            --num-instances "$num_gpus" \
                            --parallel-type "$parallel_type"
                    done
                    sleep 5
                done
                
                kill "$pid"
                sleep 5
            done
        done
    done
done