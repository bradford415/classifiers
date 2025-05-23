#!/bin/bash
mkdir imagenet
cd imagenet

# Source: https://image-net.org/challenges/LSVRC/2012/2012-downloads.php
#         use the "Training images (Task 1 & 2)"
wget https://www.image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar
wget https://www.image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar

# Below extracts the tar and splits into train and val directories
# Source: https://gist.github.com/BIGBALLON/8a71d225eff18d88e469e6ea9b39cef4
# script to extract ImageNet dataset
# ILSVRC2012_img_train.tar (about 138 GB)
# ILSVRC2012_img_val.tar (about 6.3 GB)
# make sure ILSVRC2012_img_train.tar & ILSVRC2012_img_val.tar in your current directory
#
#  https://github.com/facebook/fb.resnet.torch/blob/master/INSTALL.md
# 
#  train/
#  ├── n01440764
#  │   ├── n01440764_10026.JPEG
#  │   ├── n01440764_10027.JPEG
#  │   ├── ......
#  ├── ......
#  val/
#  ├── n01440764
#  │   ├── ILSVRC2012_val_00000293.JPEG
#  │   ├── ILSVRC2012_val_00002138.JPEG
#  │   ├── ......
#  ├── ......
#
#
# Extract the training data:
#
mkdir train && mv ILSVRC2012_img_train.tar train/ && cd train
tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar
find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
cd ..
#
# Extract the validation data and move images to subfolders:
#
mkdir val && mv ILSVRC2012_img_val.tar val/ && cd val && tar -xvf ILSVRC2012_img_val.tar
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
#
# Check total files after extract
#
#  $ find train/ -name "*.JPEG" | wc -l
#  1281167
#  $ find val/ -name "*.JPEG" | wc -l
#  50000
#