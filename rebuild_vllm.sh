# rm -r build
pip uninstall -y vllm
python3 setup.py bdist_wheel --dist-dir=dist
pip install dist/vllm-0.4.3+cu124-cp310-cp310-linux_x86_64.whl
cd benchmarks
bash 1_serving_benchmark.sh
