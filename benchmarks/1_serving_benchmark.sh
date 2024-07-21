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

scheduler_policy=(infer)
swap_policies=(partial)
# start vllm server
model_name="meta-llama/Llama-2-13b-chat-hf"
dataset_name="sharegpt"
dataset_path="/root/v1/vllm/dataset/ShareGPT_V3_unfiltered_cleaned_split.json"
result_dir="/root/v1/vllm/benchmarks/result"
preemption_mode="swap"
gpu_memory_utilization=0.5
max_num_seqs=8
swap_space=32
max_tokens=2048
iter_theshold=15
request_rates=(1000)
swap_out_partial_rates=(0.5)
gpu_devices=3

for swap_out_partial_rate in "${swap_out_partial_rates[@]}"; do
    for request_rate in "${request_rates[@]}"; do
        for swap_policy in "${swap_policies[@]}"; do
            for policy in "${scheduler_policy[@]}"; do
                CUDA_VISIBLE_DEVICES=$gpu_devices taskset -c 10-11 python3 -m vllm.entrypoints.openai.api_server \
                        --model $model_name --swap-space $swap_space\
                        --preemption-mode $preemption_mode --scheduler-policy $policy \
                        --enable-chunked-prefill --max-num-batched-tokens $max_tokens --iter-threshold $iter_theshold\
                        --max-num-seqs $max_num_seqs\
                        --swap-out-tokens-policy $swap_policy\
                        --swap-out-partial-rate $swap_out_partial_rate\
                        --gpu-memory-utilization $gpu_memory_utilization\
                        --disable-log-requests > api_server_${policy}_${swap_policy}.log 2>&1 &
                pid=$!
                
                # run benchmark and save the output to benchmark.log
                python3 benchmark_serving.py --execution-counter $COUNTER --dataset-path $dataset_path  \
                            --dataset-name $dataset_name --request-rate $request_rate \
                            --num-prompts 300 --sharegpt-output-len 2000 \
                            --model $model_name --scheduler-policy $policy \
                            --save-result --result-dir $result_dir \
                            --metadata swap_space=$swap_space preemption_mode=$preemption_mode \
                                    scheduler_policy=$policy gpu_memory_utilization=$gpu_memory_utilization \
                                    max_num_seqs=$max_num_seqs max_tokens=$max_tokens swap_policy=$swap_policy \
                                    iter_theshold=$iter_theshold swap_out_partial_rate=$swap_out_partial_rate  >> benchmark.log 2>&1
                kill $pid
                sleep 5
                python3 parse_log.py --policy $policy --swap-policy $swap_policy --result-dir $result_dir \
                            --execution-counter $COUNTER --request-rate $request_rate \
                            --swap-out-partial-rate $swap_out_partial_rate --model $model_name 
            done
        done
    done
done
