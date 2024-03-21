#!/bin/bash

# 1) Check Python version
python_version=$(python -V 2>&1 | grep -o 'Python [0-9]*.[0-9]*.[0-9]*' | cut -d ' ' -f 2)
if [ "$python_version" != "3.11.0" ]; then
    echo "Warning: Current Python version is not 3.11.0, your version is $python_version."
fi

# 2) Install torch, ask user to choose between CUDA 11.7 and CUDA 11.8
echo "Please select the version of torch to install:"
echo "Enter 117 for CUDA 11.7 or 118 for CUDA 11.8. These versions are best optimized for this project."
echo "If these are not compatible with your system, you may choose the version closest to your local environment. Most versions are generally compatible."
read -p "Enter your choice: " cuda_version

if [ "$cuda_version" == "117" ]; then
    pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117
elif [ "$cuda_version" == "118" ]; then
    pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
else
    echo "Error: Invalid input."
    exit 1
fi

# 3) Install xformers
pip install xformers==0.0.22

# 4) Install timm
pip install timm

# 5) Install mmlab related libraries
pip install -U openmim
mim install mmengine
mim install mmcv==2.0.1

# Install mmsegmentation and mmpretrain
directories=("mmsegmentation" "mmdetection")

for dir in "${directories[@]}"; do
    if [ -d "$dir" ]; then
        cd $dir
        pip install -e .
        cd ..
    else
        echo "Warning: Directory $dir does not exist. Skipping installation."
    fi
done

# 6) Install other dependencies
pip install -r requirements.txt
