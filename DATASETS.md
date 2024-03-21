# Datasets

This document outlines the datasets utilized in our project, encompassing both the training and testing sets.

## Training Datasets

### COCO
To acquire the COCO dataset, please visit [cocodataset](https://cocodataset.org/#download). The following files are required: [2017 Train Images](http://images.cocodataset.org/zips/train2017.zip), [2017 Val Images](http://images.cocodataset.org/zips/val2017.zip), and [2017 Panoptic Train/Val Annotations](http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip). These should be downloaded into the `data` directory.

Alternatively, the dataset can be downloaded using the provided script:
```shell
cd data/coco2017
bash coco2017.sh
```
The data is organized as follows:
```
data/coco2017/
├── annotations
│   ├── panoptic_train2017 [118287 entries exceeds filelimit, not opening dir]
│   ├── panoptic_train2017.json
│   ├── panoptic_val2017 [5000 entries exceeds filelimit, not opening dir]
│   └── panoptic_val2017.json
├── coco2017.sh
├── train2017 [118287 entries exceeds filelimit, not opening dir]
└── val2017 [5000 entries exceeds filelimit, not opening dir]
```


### LVIS
The LVIS dataset can be downloaded by visiting [lvisdataset](https://www.lvisdataset.org/dataset). Here, you'll find both the images and annotations.

The data is organized as follows:
```
data/lvis/
├── lvis_v1_train.json
├── lvis_v1_train.json.zip
├── lvis_v1_val.json
├── lvis_v1_val.json.zip
├── train2017 [118287 entries exceeds filelimit, not opening dir]
├── train2017.zip
├── val2017 [5000 entries exceeds filelimit, not opening dir]
└── val2017.zip
```


## Testing Datasets


### GrabCut & Berkeley & DAVIS
Please download [DAVIS](https://drive.google.com/file/d/1-ZOxk3AJXb4XYIW-7w1-AXtB9c8b3lvi/view?usp=sharing)
[GrabCut](https://drive.google.com/file/d/1CKzgFbk0guEBpewgpMUaWrM_-KSVSUyg/view?usp=sharing)
[Berkeley](https://drive.google.com/file/d/16GD6Ko3IohX8OsSHvemKG8zqY07TIm_i/view?usp=sharing) from [FocusCut](https://github.com/frazerlin/focuscut)

The data is organized as follows:
```
data/
├── berkeley
│   └── Berkeley
│       ├── gt [100 entries exceeds filelimit, not opening dir]
│       ├── img [100 entries exceeds filelimit, not opening dir]
│       └── list
│           └── val.txt
├── davis
│   └── DAVIS
│       ├── gt [345 entries exceeds filelimit, not opening dir]
│       ├── img [345 entries exceeds filelimit, not opening dir]
│       └── list
│           ├── val_ctg.txt
│           └── val.txt
└── grabcut
    └── GrabCut
        ├── gt [50 entries exceeds filelimit, not opening dir]
        ├── img [50 entries exceeds filelimit, not opening dir]
        └── list
            └── val.txt

```


### SBD
The SBD dataset is available for download at [official site](http://home.bharathh.info/pubs/codes/SBD/download.html) or [GoogleDrive](https://drive.google.com/file/d/1ISoYjyxeut3tlSGimuG3jr9RklwfnBV5/view?usp=sharing)

The data is organized as follows:
```
data/sbd/
├── benchmark_RELEASE
│   ├── dataset
│   │   ├── cls [11355 entries exceeds filelimit, not opening dir]
│   │   ├── img [11355 entries exceeds filelimit, not opening dir]
│   │   ├── inst [11355 entries exceeds filelimit, not opening dir]
│   │   ├── train.txt
│   │   └── val.txt
└── benchmark.tgz
```


### MVTec

The MVTec dataset is available for download at [Kaggle MVTec](https://www.kaggle.com/datasets/ipythonx/mvtec-ad/download?datasetVersionNumber=2)

The data is organized as follows:
```
data/mvtec
├── bottle
├── cable
├── capsule
├── carpet
├── grid
├── hazelnut
├── leather
├── metal_nut
├── pill
├── screw
├── tile
├── toothbrush
├── transistor
├── wood
└── zipper
```

### COD10K

The COD10K dataset is available for download at [COD10K](https://drive.google.com/file/d/1vRYAie0JcNStcSwagmCq55eirGyMYGm5/view?usp=sharing)

The data is organized as follows:
```
data/COD10K-v3
├── Info
│   ├── CAM_test.txt
│   ├── CAM_train.txt
│   ├── NonCAM_test.txt
│   ├── NonCAM_train.txt
│   └── Statistics-CAM.xlsx
├── Readme.txt
├── Test
│   ├── CAM_Instance_Test.json
│   ├── CAM-NonCAM_Instance_Test.txt
│   ├── GT_Edge
│   ├── GT_Instance
│   ├── GT_Object
│   └── Image
└── Train
    ├── CAM_Instance_Train.json
    ├── CAM-NonCAM_Instance_Train.txt
    ├── GT_Edge
    ├── GT_Instance
    ├── GT_Object
    └── Image
```
