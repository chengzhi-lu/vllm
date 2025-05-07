gpu_devices=$1
model_name=$2
swap_space=$3
preemption_mode=$4
policy=$5
max_tokens=$6
iter_theshold=$7
max_num_seqs=$8
swap_policy=$9
swap_out_partial_rate=${10}
gpu_memory_utilization=${11}
waiting_iter=${12}


CUDA_VISIBLE_DEVICES=$gpu_devices taskset -c 10-11 python3 -m vllm.entrypoints.openai.api_server \
    --model $model_name --swap-space $swap_space --preemption-mode $preemption_mode --scheduler-policy $policy \
    --enable-chunked-prefill --max-num-batched-tokens $max_tokens --iter-threshold $iter_theshold --max-num-seqs $max_num_seqs --swap-out-tokens-policy $swap_policy --swap-out-partial-rate $swap_out_partial_rate --execution-budget $iter_theshold \
    --gpu-memory-utilization $gpu_memory_utilization --waiting-iter-base $waiting_iter --disable-log-requests >api_server_${policy}_${swap_policy}.log 2>&1
