#!/bin/bash

cd ../
#python3 train_search.py --criterion=ls --unroll;
#python3 train_search.py --criterion=ce --unroll;
#python3 train_search.py --criterion=softece --unroll;
#python3 train_search.py --criterion=mmce --unroll;
#python3 train_search.py --criterion=klece --unroll
#CUDA_VISIBLE_DEVICES=0 python3 train_search.py --criterion=softece --unroll --auxloss_coef=0.1 &
#CUDA_VISIBLE_DEVICES=1 python3 train_search.py --criterion=mmce --unroll --auxloss_coef=100;
CUDA_VISIBLE_DEVICES=1 python3 train_search.py --criterion=ce
