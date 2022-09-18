#!/bin/bash

cd /home/linwei/Desktop/projects/NAS-Calibration
for a in  'wide_resnet101_2' 'wide_resnet50_2' 'squeezenet1_0' 'squeezenet1_1' 'mnasnet0_5' 'mnasnet0_75' 'mnasnet1_0' 'mnasnet1_3' 'shufflenet_v2_x0_5' 'shufflenet_v2_x1_0''shufflenet_v2_x1_5' 'shufflenet_v2_x2_0' 'inception_v3'
do
   CUDA_VISIBLE_DEVICES=0 python3 train.py --drop_path_prob=0 --weight_decay=5e-4 --epochs=350 --scheduler=focal --batch_size=128 --arch=$a --criterion=focal --focal_gamma=3
done