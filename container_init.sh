apt update
apt install -y curl
curl https://sh.rustup.rs -sSf | sh
source $HOME/.cargo/env
pip install --upgrade pip
pip install -r requirements-common.txt
