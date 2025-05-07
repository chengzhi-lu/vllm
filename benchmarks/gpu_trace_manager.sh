#!/bin/bash

start_trace(){
    local file_name=$1
    dcgmi dmon -e 1002,1003,1005,1009  -d 500 > $file_name 2>&1 &
    echo "Trace started. Trace file: $file_name"
}


stop_trace(){
    kill -9 `ps -aux|grep dcgmi |awk '{print $2}'`
    echo "Trace stopped."
}

if [ "$1" == "start" ]; then
    start_trace $2
elif [ "$1" == "stop" ]; then
    stop_trace
else
    echo "Usage: $0 start <trace_file_name> | stop"
fi