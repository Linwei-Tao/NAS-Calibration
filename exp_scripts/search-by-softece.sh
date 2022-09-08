#!/bin/bash
# lab2
cd ../
CUDA_VISIBLE_DEVICES=0 python3 train_search.py --criterion=softece --unroll;