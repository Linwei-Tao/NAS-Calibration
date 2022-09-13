#!/bin/bash

cd ../
CUDA_VISIBLE_DEVICES=1 /home/linwei/anaconda3/bin/python3 train.py --drop_path_prob=0 --weight_decay=5e-4 --epochs=350 --scheduler=focal --batch_size=128 --arch=DARTS_SOFTECE01 --save=retrain --criterion=softece &



