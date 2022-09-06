#!/bin/bash

cd ../
CUDA_VISIBLE_DEVICES=0 python3 train.py --drop_path_prob=0 --weight_decay=5e-4 --epochs=350 --scheduler=focal --batch_size=128 --save=ablation1 &
CUDA_VISIBLE_DEVICES=1 python3 train.py --drop_path_prob=0 --weight_decay=3e-4 --epochs=350 --scheduler=focal --batch_size=128 --save=ablation2 ;
CUDA_VISIBLE_DEVICES=0 python3 train.py --drop_path_prob=0 --weight_decay=3e-4 --epochs=350 --scheduler=focal --batch_size=128 --auxiliary --save=ablation3 &
CUDA_VISIBLE_DEVICES=1 python3 train.py --drop_path_prob=0 --weight_decay=3e-4 --epochs=350 --scheduler=focal --batch_size=96 --auxiliary --save=ablation4 ;
CUDA_VISIBLE_DEVICES=0 python3 train.py --drop_path_prob=0.2 --weight_decay=3e-4 --epochs=350 --scheduler=focal --batch_size=96 --auxiliary --save=ablation5 &
CUDA_VISIBLE_DEVICES=1 python3 train.py --drop_path_prob=0.2 --weight_decay=3e-4 --epochs=600 --scheduler=darts --batch_size=96 --auxiliary --save=ablation6


