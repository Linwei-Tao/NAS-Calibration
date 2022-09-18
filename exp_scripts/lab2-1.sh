#!/bin/bash

cd /home/linwei/Desktop/projects/NAS-Calibration
for a in 'alexnet' 'resnet18' 'resnet34' 'resnet50' 'resnet101' 'resnet152' 'wide_resnet101_2' 'wide_resnet50_2'
do
   CUDA_VISIBLE_DEVICES=1 python3 train.py --drop_path_prob=0 --weight_decay=5e-4 --epochs=350 --scheduler=focal --batch_size=128 --arch=$a --criterion=focal --focal_gamma=3
done
