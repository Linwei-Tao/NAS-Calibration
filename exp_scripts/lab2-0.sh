##!/bin/bash
#
#cd /home/linwei/Desktop/projects/NAS-Calibration
#for a in 'resnext50_32x4d' 'resnext101_32x8d' 'densenet121' 'densenet161' 'densenet169' 'densenet201'
#do
#   CUDA_VISIBLE_DEVICES=0 python3 train.py --drop_path_prob=0 --weight_decay=5e-4 --epochs=350 --scheduler=focal --batch_size=128 --arch=$a --criterion=focal --focal_gamma=3
#done



cd /home/linwei/Desktop/projects/NAS-Calibration

CUDA_VISIBLE_DEVICES=0 python3 train.py --drop_path_prob=0 --weight_decay=5e-4 --epochs=350 --scheduler=focal --batch_size=128 --arch=resnet50 --criterion=focal --focal_gamma=3 --grad_clip=2 &
CUDA_VISIBLE_DEVICES=1 python3 train.py --drop_path_prob=0 --weight_decay=5e-4 --epochs=350 --scheduler=focal --batch_size=128 --arch=resnet50_pytorch --criterion=focal --focal_gamma=3 --grad_clip=2


