
start_ray() {
  # 尝试获取ray状态
  if ray status &> /dev/null; then
    # 如果ray已经在运行，则先停止
    echo "Ray cluster is already running, stopping it first..."
    stop_ray
  fi
  # 无论之前是否运行，现在都启动ray
  echo "Starting Ray cluster..."
  NCCL_SOCKET_IFNAME=bond0 RAY_ENABLE_RECORD_ACTOR_TASK_LOGGING=1 ray start --head
  ssh lucz@10.119.46.53 'docker exec -i vllm_lucz bash -c "NCCL_SOCKET_IFNAME=bond0 RAY_ENABLE_RECORD_ACTOR_TASK_LOGGING=1 ray start --address=10.119.46.54:6379"'
}

stop_ray() {
  ray stop
  ssh lucz@10.119.46.53 'docker exec -i vllm_lucz bash -c "bash /root/vllm/benchmarks/terminate_server.sh"' || true
  rm -r /tmp/ray
  sleep 5
}
start_server() {
  local policy=$1
  local swap_policy=$2
  local swap_out_partial_rate=$3
  local parallel_type=$4
  local model_name=$5

  local log_file="logs/api_server_${model_name##*/}_${parallel_type}_${policy}_${swap_policy}.log"
  
  local parallel_args=""
  # GPU分配逻辑
  if [[ "$parallel_type" == @("pp"|"tp") ]]; then
      gpu_devices="0,1,2,3"
  elif [[ "$parallel_type" == "single" && "$model_name" == "meta-llama/Llama-2-13b-chat-hf" ]]; then
      gpu_devices="0"
  elif [[ "$parallel_type" == "single" && "$model_name" == "meta-llama/Llama-2-70b-chat-hf" ]]; then
      echo "70b模型不支持单卡"
      return 1  # 添加返回语句避免继续执行
  else
      gpu_devices="0,1,2,3"
  fi
  # 如果是 pp，则调用 ray start
  if [[ "$parallel_type" == "pp" ]]; then
      start_ray
  fi

  case "$policy" in
    "tfittradeoff")
        max_num_seqs=192
        ;;
    *)
        max_num_seqs=128
        ;;
  esac
  echo "启动服务器: model=${model_name##*/} type=$parallel_type policy=$policy batch size=$max_num_seqs"
  IFS=',' read -ra gpu_array <<< "$gpu_devices"
  num_gpus=${#gpu_array[@]}

  case "$parallel_type" in
    "single")
      parallel_args="--tensor-parallel-size $num_gpus"
      ;;
    "pp")
      parallel_args="--pipeline-parallel-size 8 --worker-use-ray --host $host"
      ;;
    "tp")
      parallel_args="--tensor-parallel-size $num_gpus --worker-use-ray"
      ;;
  esac
  prefill_predictor_model_config=""
  if [[ "$policy" == "opt" ]]; then
      case "$model_name" in 
        "meta-llama/Llama-2-13b-chat-hf")
          prefill_predictor_model_config_path="/root/vllm/train/MODEL/results/opt-125m-llama2-13b-sharegpt-score-trainbucket10-b32/usage_config.json"
          ;;
        "meta-llama/Llama-2-13b-chat-hf")
          prefill_predictor_model_config_path="/root/vllm/train/MODEL/results/opt-350m-llama2-70b-sharegpt-score-trainbucket10-b32/usage_config.json"
          ;;
      esac
      if [[ "$prefill_predictor_model_config" == "" ]]; then
          echo "prefill_predictor_model_config: $prefill_predictor_model_config_path"
          prefill_predictor_model_config="--prefill-predictor-model-config $prefill_predictor_model_config_path"
      fi
  fi


  CUDA_VISIBLE_DEVICES=$gpu_devices RAY_DEDUP_LOGS=0 taskset -c 28-29 python3 -m vllm.entrypoints.openai.api_server \
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
    --max-serving-time "$max_serving_time" \
    $prefill_predictor_model_config > "$log_file" 2>&1 &
  
  if [[ "$parallel_type" == @("pp") ]]; then
      sleep 5
      remote_shell="bash /root/vllm/benchmarks/3_serving_benchmark_pp.sh \"$model_name\" \"$max_num_seqs\" \"$policy\" \"$swap_policy\" \"$swap_out_partial_rate\""
      ssh lucz@10.119.46.53 \
        "docker exec -i vllm_lucz bash -c \"$remote_shell\""
  fi
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
  local total_request_nums=$9

  local request_duration=$((total_request_nums / request_rate))
  local log_file="logs/benchmark_${model_name##*/}_${parallel_type}_${policy}_rate${request_rate}.log"

  echo "运行基准测试: model=${model_name##*/} rate=$request_rate"
  taskset -c 30-49 python3 benchmark_serving.py \
    --execution-counter "$COUNTER" \
    --dataset-path "$dataset_path" \
    --dataset-name "$dataset_name" \
    --request-rate "$request_rate" \
    --num-prompts 8192 \
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
}


terminate_server() {
  local parallel_type=$1
  kill -9 `ps -aux|grep "vllm.entrypoints.openai.api_server" | grep -v grep | awk '{print $2}'` 2>/dev/null
  if [[ "$parallel_type" == @("pp"|"tp") ]]; then
      kill -9 `ps -aux|grep "ray" | grep -v grep | awk '{print $2}'` 2>/dev/null
      if [[ "$parallel_type" == "pp" ]]; then
          ssh lucz@10.119.46.53 'docker exec -i vllm_lucz bash -c "bash /root/vllm/benchmarks/terminate_server.sh"'
          stop_ray
      fi
  fi
}
