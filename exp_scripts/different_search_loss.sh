#!/bin/bash

cd ../
python3 train_search.py --criterion=ls --unroll;
python3 train_search.py --criterion=ce --unroll;
python3 train_search.py --criterion=softece --unroll;
python3 train_search.py --criterion=mmce --unroll;