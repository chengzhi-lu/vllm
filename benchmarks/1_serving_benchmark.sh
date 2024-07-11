# 读取当前计数器的值
COUNTER_FILE=".counter.txt"
if [ -f "$COUNTER_FILE" ]; then
  COUNTER=$(cat $COUNTER_FILE)
else
  COUNTER=0
fi
# 自增计数器
COUNTER=$((COUNTER + 1))
# 将新的计数器值写回文件
echo $COUNTER > $COUNTER_FILE

scheduler_policy=(infer fcfs)
swap_policies=(partial full)
# start vllm server
model_name="meta-llama/Llama-2-13b-chat-hf"
dataset_name="sharegpt"
dataset_path="/root/vllm/dataset/ShareGPT_V3_unfiltered_cleaned_split.json"
result_dir="/root/vllm/benchmarks/result"
preemption_mode="swap"
gpu_memory_utilization=0.5
max_num_seqs=64
swap_space=32
max_tokens=2048
iter_theshold=10
request_rates=(2 4 8 12 16)
swap_out_partial_rates=(0.1 0.2 0.3 0.4 0.5)
for request_rate in "${request_rates[@]}"; do
    for swap_policy in "${swap_policies[@]}"; do
        for policy in "${scheduler_policy[@]}"; do
            for swap_out_partial_rate in "${swap_out_partial_rates[@]}"; do
                if [ $swap_policy == "partial" ]; then
                    CUDA_VISIBLE_DEVICES=1 python3 -m vllm.entrypoints.openai.api_server \
                            --model $model_name --swap-space $swap_space\
                            --preemption-mode $preemption_mode --scheduler-policy $policy \
                            --enable-chunked-prefill --max-num-batched-tokens $max_tokens\
                            --max-num-seqs $max_num_seqs\
                            --swap-out-partial-tokens\
                            --swap-out-partial-rate $swap_out_partial_rate\
                            --gpu-memory-utilization $gpu_memory_utilization\
                            --disable-log-requests > api_server_${policy}_${swap_policy}.log 2>&1 &
                    pid=$!
                else
                    CUDA_VISIBLE_DEVICES=1 taskset -c 10-11 python3 -m vllm.entrypoints.openai.api_server \
                            --model $model_name --swap-space $swap_space\
                            --preemption-mode $preemption_mode --scheduler-policy $policy \
                            --enable-chunked-prefill --max-num-batched-tokens $max_tokens\
                            --max-num-seqs $max_num_seqs\
                            --gpu-memory-utilization $gpu_memory_utilization\
                            --disable-log-requests > api_server_${policy}_${swap_policy}.log 2>&1 &
                    pid=$!
                fi
                # run benchmark and save the output to benchmark.log
                python3 benchmark_serving.py --execution-counter $COUNTER --dataset-path $dataset_path  \
                            --dataset-name $dataset_name --request-rate $request_rate \
                            --num-prompts 300 --sharegpt-output-len 1000 \
                            --model $model_name --scheduler-policy $policy \
                            --save-result --result-dir $result_dir \
                            --metadata swap_space=$swap_space preemption_mode=$preemption_mode \
                                    scheduler_policy=$policy gpu_memory_utilization=$gpu_memory_utilization \
                                    max_num_seqs=$max_num_seqs max_tokens=$max_tokens swap_policy=$swap_policy \
                                    iter_theshold=$iter_theshold swap_out_partial_rate=$swap_out_partial_rate  >> benchmark.log 2>&1
                kill $pid
                sleep 5
            done
        done
    done
done
