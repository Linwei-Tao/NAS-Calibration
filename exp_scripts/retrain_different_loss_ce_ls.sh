#!/bin/bash

cd ../
CUDA_VISIBLE_DEVICES=0 python3 train.py --drop_path_prob=0 --weight_decay=5e-4 --epochs=350 --scheduler=focal --batch_size=128 --save=retrain_ce --arch=DARTS_CE;
CUDA_VISIBLE_DEVICES=0 python3 train.py --drop_path_prob=0 --weight_decay=5e-4 --epochs=350 --scheduler=focal --batch_size=128 --save=retrain_ls --arch=DARTS_LS


