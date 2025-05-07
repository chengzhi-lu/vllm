terminate_server() {
  local parallel_type=$1
  ray stop
  kill -9 `ps -aux|grep "vllm.entrypoints.openai.api_server" | grep -v grep | awk '{print $2}'` 2>/dev/null
  if [[ "$parallel_type" == @("pp"|"tp") ]]; then
      kill -9 `ps -aux|grep "ray" | grep -v grep | awk '{print $2}'` 2>/dev/null
  fi
}

parallel_type="pp"
terminate_server $parallel_type