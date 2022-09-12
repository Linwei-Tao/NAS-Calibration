#!/bin/bash

cd ../
CUDA_VISIBLE_DEVICES=1 python3 train_search.py --criterion=softece --auxloss_coef=1
CUDA_VISIBLE_DEVICES=1 python3 train_search.py --criterion=softece --auxloss_coef=5
CUDA_VISIBLE_DEVICES=1 python3 train_search.py --criterion=softece --auxloss_coef=10
