# Installation Guide

This guide provides detailed steps for setting up the environment required to run the project. Please follow the instructions according to your system's specifications.

## Prerequisites
- Python 3.11.0

## PyTorch Installation
- For CUDA 11.7
```shell
pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117
```
- For CUDA 11.8
```shell
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
```
- For CPU version
```shell
pip install torch==2.0.1+cpu torchvision==0.15.2+cpu torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu
```
Additional Libraries
- xformers (For GPU environment, Optional)
```shell
pip install xformers==0.0.22
```
- mmcv 2.0.1
```shell
pip install -U openmim
mim install mmengine==0.9.0
mim install mmcv==2.0.1
cd mmsegmentation
mim install -e .
cd ../mmdetection
mim install -e .
cd ..
```
- Other dependencies
```shell
pip install -r requirements.txt
```
