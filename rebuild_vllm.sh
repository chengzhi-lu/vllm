# rm -r build
pip uninstall -y vllm
taskset -c 35-40 python3 setup.py bdist_wheel --dist-dir=dist
pip install dist/vllm-0.5.2+cu124-cp310-cp310-linux_x86_64.whl
pip uninstall -y vllm-flash-attn
clear
cd benchmarks
bash 1_serving_benchmark.sh
# bash motivation_benchmark.sh
