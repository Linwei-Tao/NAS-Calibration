#!/bin/bash

cd ../
CUDA_VISIBLE_DEVICES=0 python3 train_search.py --criterion=softece --auxloss_coef=0.01
CUDA_VISIBLE_DEVICES=0 python3 train_search.py --criterion=softece --auxloss_coef=0.1
CUDA_VISIBLE_DEVICES=0 python3 train_search.py --criterion=softece --auxloss_coef=0.5
