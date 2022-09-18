#!/bin/bash


## train with focal
#ssh ltao0358@hpc.sydney.edu.au 'cd /scratch/ContraGAN/projects/NAS-Calibration/exp_scripts/USYD-HPC-Scripts/;
#qsub -v ARCH="DARTS_MMCE1" retrain.sh
#qsub -v ARCH="DARTS_MMCE10" retrain.sh
#qsub -v ARCH="DARTS_MMCE100" retrain.sh
#qsub -v ARCH="DARTS_MMCE1000" retrain.sh
#qsub -v ARCH="DARTS_SOFTECE001" retrain.sh
#qsub -v ARCH="DARTS_SOFTECE01" retrain.sh
#qsub -v ARCH="DARTS_SOFTECE05" retrain.sh
#qsub -v ARCH="DARTS_SOFTECE1" retrain.sh
#qsub -v ARCH="DARTS_SOFTECE5" retrain.sh
#qsub -v ARCH="DARTS_SOFTECE10" retrain.sh
#qsub -v ARCH="DARTS_CE" retrain.sh
#'

# train with focal
ssh ltao0358@hpc.sydney.edu.au 'cd /scratch/ContraGAN/projects/NAS-Calibration/exp_scripts/USYD-HPC-Scripts/retrain_MMCE;
qsub mmce1
qsub mmce10
qsub mmce100
qsub mmce1000
'