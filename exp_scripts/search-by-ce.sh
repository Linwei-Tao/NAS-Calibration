#!/bin/bash

cd ../
CUDA_VISIBLE_DEVICES=0 python3 train_search.py --criterion=ce --unroll;