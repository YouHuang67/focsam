#!/bin/sh
echo "Prepare to download train2017 image zip file..."
wget -c http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip

echo "Prepare to download test2017 image zip file..."
wget -c http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip

echo "Prepare to download train-val2017 panoptic annotations zip file..."
wget -c http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip
unzip panoptic_annotations_trainval2017.zip
unzip annotations/panoptic_train2017.zip -d annotations
unzip annotations/panoptic_val2017.zip -d annotations
