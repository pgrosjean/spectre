# This script uses CUDA 12.1. You can swap with CUDA 11.8.
mamba create --name spectre \
    python=3.10 \
    pytorch-cuda=12.1 \
    pytorch=2.3.0 cudatoolkit xformers torchvision -c pytorch -c nvidia -c xformers \
    -y
conda activate spectre

pip install -e . -v
