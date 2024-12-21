#!/bin/bash

for i in {0..30}
do
    echo "start $i ..."
    bash rebuild_vllm.sh
    echo "finish $i!"
    sleep 30
done
