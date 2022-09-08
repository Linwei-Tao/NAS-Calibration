#!/bin/bash

cd ../
CUDA_VISIBLE_DEVICES=1 python3 train_search.py --criterion=mmce --unroll;