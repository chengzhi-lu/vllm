#!/bin/bash

for i in {0..2}
do
    echo "start $i ..."
    bash rebuild_vllm.sh
    echo "finish $i!"
done
