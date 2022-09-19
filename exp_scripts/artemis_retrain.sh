#!/bin/bash

# train with focal
ssh ltao0358@hpc.sydney.edu.au 'cd /scratch/ContraGAN/projects/NAS-Calibration/exp_scripts/USYD-HPC-Scripts/;
qsub -v ARCH="DARTS_MMCE1",CRITERION="focal",COEF=1 retrain.sh
qsub -v ARCH="DARTS_MMCE10",CRITERION="focal",COEF=1 retrain.sh
qsub -v ARCH="DARTS_MMCE100",CRITERION="focal",COEF=1 retrain.sh
qsub -v ARCH="DARTS_MMCE1000",CRITERION="focal",COEF=1 retrain.sh
qsub -v ARCH="DARTS_SOFTECE001",CRITERION="focal",COEF=1 retrain.sh
qsub -v ARCH="DARTS_SOFTECE01",CRITERION="focal",COEF=1 retrain.sh
qsub -v ARCH="DARTS_SOFTECE05",CRITERION="focal",COEF=1 retrain.sh
qsub -v ARCH="DARTS_SOFTECE1",CRITERION="focal",COEF=1 retrain.sh
qsub -v ARCH="DARTS_SOFTECE5",CRITERION="focal",COEF=1 retrain.sh
qsub -v ARCH="DARTS_SOFTECE10",CRITERION="focal",COEF=1 retrain.sh
qsub -v ARCH="DARTS_CE",CRITERION="focal",COEF=1 retrain.sh
qsub -v ARCH="DARTS_MMCE1",CRITERION="mmce",COEF=8 retrain.sh
qsub -v ARCH="DARTS_MMCE10",CRITERION="mmce",COEF=8 retrain.sh
qsub -v ARCH="DARTS_MMCE100",CRITERION="mmce",COEF=8 retrain.sh
qsub -v ARCH="DARTS_MMCE100",CRITERION="mmce",COEF=8 retrain.sh
qsub -v ARCH="DARTS_MMCE1",CRITERION="mmce",COEF=9 retrain.sh
qsub -v ARCH="DARTS_MMCE10",CRITERION="mmce",COEF=9 retrain.sh
qsub -v ARCH="DARTS_MMCE100",CRITERION="mmce",COEF=9 retrain.sh
qsub -v ARCH="DARTS_MMCE100",CRITERION="mmce",COEF=9 retrain.sh
'