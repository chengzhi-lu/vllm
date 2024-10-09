#!/bin/bash

for i in {1..3}
do
    echo "first $i ..."
    bash rebuild_vllm.sh
    echo "finish $i!"
done
