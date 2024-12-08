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
echo $COUNTER >$COUNTER_FILE

# start vllm server
pwd=`pwd`
# model_name="meta-llama/Llama-2-70b-chat-hf"
model_name="meta-llama/Llama-2-13b-chat-hf"
# model_name="mistralai/Mistral-7B-Instruct-v0.1" # 32000
# model_name="EleutherAI/gpt-neox-20b"
# model_name="facebook/opt-6.7b"
dataset_name="sharegpt"
dataset_path="/root/vllm/dataset/ShareGPT_V3_unfiltered_cleaned_split.json"
result_dir="${pwd}/result"
# scheduler_policy=(fcfs)
# swap_policies=(full)
# scheduler_policy=(infer)
# swap_policies=(partial)
declare -a scheduler_swap_policies
scheduler_swap_policies[0]="tfittradeoff partial"
scheduler_swap_policies[1]="fcfs full"
scheduler_swap_policies[2]="las full"
# scheduler_swap_policies[3]="tfittradeoff full"
scheduler_swap_policies[4]="sjf full"
scheduler_swap_policies[5]="srjf full"
# scheduler_swap_policies[3]="sjmlfq full"
# scheduler_swap_policies[3]="infer partial"
# scheduler_swap_policies[4]="inferpreempt full"
# scheduler_swap_policies[5]="sjmlfq full"fish

preemption_mode="swap"
gpu_memory_utilization=0.7 # 0.5, 0.7, 0.9
max_num_seqs=384
# max_num_seqs=1024
swap_space=64
# swap_space=32
max_tokens=2048
# max_tokens=4096
iter_theshold=15
max_serving_time=86400 # 86400
request_duration=300 # 1
num_shared_blocks=0

# request_rates[0]=0.5
request_rates[3]=1.0
request_rates[2]=2.0
request_rates[1]=5.0
request_rates[0]=10.0
# request_rates[4]=10.0
# request_rates[5]=20.0
# request_rates[5]=50.0
# request_rates[5]=30.0
# request_rates[5]=50.0
# request_rates[5]=100.0

# request_rates=(2.0)
swap_out_partial_rates=(0.5)
waiting_iter_base=(0.1)
gpu_devices=0

for waiting_iter in "${waiting_iter_base[@]}"; do
  for swap_out_partial_rate in "${swap_out_partial_rates[@]}"; do
    for scheduler_swap_policy in "${scheduler_swap_policies[@]}"; do
      element=(${scheduler_swap_policy})
      policy=${element[0]}
      swap_policy=${element[1]}

      CUDA_VISIBLE_DEVICES=$gpu_devices taskset -c 28-29 python3 -m vllm.entrypoints.openai.api_server \
      --model $model_name --swap-space $swap_space --preemption-mode $preemption_mode --scheduler-policy $policy \
      --enable-chunked-prefill --max-num-batched-tokens $max_tokens --iter-threshold $iter_theshold --max-num-seqs $max_num_seqs --swap-out-tokens-policy $swap_policy --swap-out-partial-rate $swap_out_partial_rate --execution-budget $iter_theshold \
      --tensor-parallel-size 1 --num-shared-blocks $num_shared_blocks --gpu-memory-utilization $gpu_memory_utilization --disable-sliding-window --waiting-iter-base $waiting_iter --disable-log-requests --max-serving-time $max_serving_time >api_server_${policy}_${swap_policy}.log 2>&1 &
      pid=$!

        for request_rate in "${request_rates[@]}"; do
          for i in {0..0}; do
            taskset -c 30-49 python3 benchmark_serving.py --execution-counter $COUNTER --dataset-path $dataset_path \
              --dataset-name $dataset_name --request-rate $request_rate \
              --num-prompts 3000 --request-duration $request_duration --sharegpt-output-len 2000 --model $model_name --scheduler-policy $policy \
              --save-result --result-dir $result_dir \
              --metadata swap_space=$swap_space preemption_mode=$preemption_mode \
              scheduler_policy=$policy gpu_memory_utilization=$gpu_memory_utilization \
              max_num_seqs=$max_num_seqs max_tokens=$max_tokens swap_policy=$swap_policy \
              iter_theshold=$iter_theshold swap_out_partial_rate=$swap_out_partial_rate waiting_iter_base=$waiting_iter \
              >> benchmark-${policy}.log 2>&1
            
            sleep 5
            python3 parse_log.py --policy $policy --swap-policy $swap_policy --result-dir $result_dir \
              --execution-counter $COUNTER --request-rate $request_rate \
              --swap-out-partial-rate $swap_out_partial_rate --model $model_name
          done
          sleep 120
        done
      kill $pid
      sleep 5
    done
  done
done
