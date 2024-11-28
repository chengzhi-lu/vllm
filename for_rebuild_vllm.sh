#!/bin/bash

for i in {1..2}
do
    echo "start $i ..."
    bash rebuild_vllm.sh
    echo "finish $i!"
done
