#!/bin/bash

for i in {1..3}
do
    echo "start $i ..."
    bash rebuild_vllm.sh
    echo "finish $i!"
done
