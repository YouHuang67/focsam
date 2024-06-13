<div align="center">
  <img src="resources/logo.png" width="400"/>
  </div>
  <div>&nbsp;</div>

# Introduction

This repository contains the implementation for
> [FocSAM: Delving Deeply into Focused Objects in Segmenting Anything](https://arxiv.org/abs/2405.18706)


# Demo
The following GIF animations display a comparison of interactive segmentation results between SAM and our FocSAM. Notably, FocSAM demonstrates a remarkably stable performance with significantly less fluctuation in IoU compared to SAM, across various datasets.


<img src="resources/result-2008_002715.gif" width="250" height="250"/><img src="resources/result-2009_002177.gif" width="250" height="250"/><img src="resources/result-2009_004203.gif" width="250" height="250"/> 


<img src="resources/result-2010_000197.gif" width="250" height="250"/><img src="resources/result-cable_cut_inner_insulation_007.gif" width="250" height="250"/><img src="resources/result-capsule_squeeze_004.gif" width="250" height="250"/> 


<img src="resources/result-COD10K-CAM-1-Aquatic-13-Pipefish-836.gif" width="250" height="250"/><img src="resources/result-COD10K-CAM-3-Flying-53-Bird-3089.gif" width="250" height="250"/><img src="resources/result-COD10K-CAM-3-Flying-53-Bird-3141.gif" width="250" height="250"/>


<img src="resources/result-grid_bent_005.gif" width="250" height="250"/><img src="resources/result-transistor_bent_lead_007.gif" width="250" height="250"/><img src="resources/result-zipper_combined_000.gif" width="250" height="250"/>


# Installation

For detailed installation instructions, please refer to [INSTALL](INSTALL.md).

Alternatively, ensure you have Python version 3.11.0 set up in your environment. Then, install all dependencies by running the following command in your terminal:

```bash
bash scripts/install.sh
```

# Dataset Preparation

For detailed dataset preparation instructions, please refer to [DATASETS](DATASETS.md).


# Model Weights Download and Conversion

## SAM Pre-trained Weights

- Download: Acquire the pretrained [SAM-ViT-H](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) and save it to `pretrain/sam_vit_h_4b8939.pth`.
- Conversion: Convert the downloaded weights using the command below:
```shell
python tools/model_converters/samvit2mmclickseg.py pretrain/sam_pretrain_vit_huge.pth
```

## FocSAM Pre-trained Weights
- Download: Obtain the pretrained [FocSAM-ViT-H](https://drive.google.com/file/d/1VOeMkY9LovfWYi66JMR8q79gFiC4n3Az/view?usp=sharing), and unzip it in `work_dirs/focsam/focsam_vit_huge_eval`.

# Evaluating the Model
- Single GPU (Example for DAVIS dataset):
```shell
export PYTHONPATH=.
python tools/test_no_viz.py configs/_base_/eval_davis.py work_dirs/focsam/focsam_vit_huge_eval/iter_160000.pth
```
- Multi-GPU:
```shell
bash tools/dist_test.sh configs/_base_/eval_davis.py work_dirs/focsam/focsam_vit_huge_eval/iter_160000.pth 4
```
- CPU (Not recommended):
```shell
export PYTHONPATH=.
CUDA_VISIBLE_DEVICES= python tools/test_no_viz.py configs/_base_/eval_davis.py work_dirs/focsam/focsam_vit_huge_eval/iter_160000.pth
```
- Evaluating on Other Datasets: Replace the config file for other datasets as needed:
```shell
configs/_base_/eval_sbd.py  # for SBD
configs/_base_/eval_grabcut.py  # for GrabCut 
configs/_base_/eval_berkeley.py  # for Berkeley
configs/_base_/eval_mvtec.py  # for MVTec
configs/_base_/eval_cod10k.py  # for COD10K
```

# Training the Model

## Training SAM Decoder
- Single GPU:
```shell
export PYTHONPATH=.
python tools/train.py configs/sam/coco_lvis/train_colaug_coco_lvis_1024x1024_320k.py
```
- Multi-GPU:
```shell
bash tools/dist_train.sh configs/sam/coco_lvis/train_colaug_coco_lvis_1024x1024_320k.py 4
```
- CPU (Not recommended):
```shell
export PYTHONPATH=.
CUDA_VISIBLE_DEVICES= python tools/train.py configs/sam/coco_lvis/train_colaug_coco_lvis_1024x1024_320k.py
```

## Training FocSAM Refiner
- Important Pre-requisite: Begin by training the SAM decoder. This step produces the required file `work_dirs/sam/coco_lvis/train_colaug_coco_lvis_1024x1024_320k/iter_320000.pth`, which is essential for the subsequent training of the FocSAM refiner.

- Single GPU:
```shell
export PYTHONPATH=.
python tools/train.py configs/focsam/coco_lvis/train_colaug_coco_lvis_1024x1024_160k.py
```
- Multi-GPU:
```shell
bash tools/dist_train.sh configs/focsam/coco_lvis/train_colaug_coco_lvis_1024x1024_160k.py 4
```
- CPU (Not recommended):
```shell
export PYTHONPATH=.
CUDA_VISIBLE_DEVICES= python tools/train.py configs/focsam/coco_lvis/train_colaug_coco_lvis_1024x1024_160k.py
```

# License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
