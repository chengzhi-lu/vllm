# list scheduler_policy
scheduler_policy=(fcfs infer)
# start vllm server
model_name="meta-llama/Llama-2-13b-chat-hf"
dataset_name="sharegpt"
dataset_path="/root/vllm/dataset/ShareGPT_V3_unfiltered_cleaned_split.json"
result_dir="/root/vllm/benchmarks/result"
for policy in "${scheduler_policy[@]}"; do
    python3 -m vllm.entrypoints.openai.api_server \
            --model $model_name --swap-space 16 \
            --preemption-mode swap --scheduler-policy $policy \
            --enable-chunked-prefill --max-num-batched-tokens 2048 \
            --max-num-seqs 128 \
            --disable-log-requests >> api_server.log 2>&1 &
    pid=$!
    sleep 20
    # run benchmark and save the output to benchmark.log
    python3 benchmark_serving.py --dataset-path $dataset_path  --dataset-name $dataset_name --request-rate 1000 --num-prompts 1000 --sharegpt-output-len 1000 --model $model_name --result-dir $result_dir --scheduler-policy $policy --save-result >> benchmark.log 2>&1
    kill $pid
done