#!/bin/bash

# For reference see the following links:
# https://aws-dglke.readthedocs.io/en/latest/train.html
# https://github.com/awslabs/dgl-ke/blob/master/notebook-examples/kge_wikimedia.ipynb

EXPERIMENT_NAME="drkg"
FORMAT="raw_udd_hrt"
# MODEL
MODEL_NAME="TransE_l2"
GAMMA=12  # For translational models
LR=0.1
HIDDEN_DIM=200
# TRAINING
MAX_STEP=2000
BATCH_SIZE=1200
NEG_SAMPLE_SIZE=200
LOG_INTERVAL=1000
###
BATCH_SIZE_EVAL=16
NEG_SAMPLE_SIZE_EVAL=10000
REGULARIZATION_COEF=1.00E-09
# HARDWARE
GPU=0
NUM_THREADS=7


dglke_train --dataset $EXPERIMENT_NAME --data_path $EXPERIMENT_NAME$"/" --save_path $EXPERIMENT_NAME$"/" \
--model_name $MODEL_NAME --data_files "train.txt" "valid.txt" "test.txt" --format $FORMAT \
--lr $LR --max_step $MAX_STEP --log_interval $LOG_INTERVAL \
--hidden_dim $HIDDEN_DIM \
--batch_size $BATCH_SIZE --neg_sample_size $NEG_SAMPLE_SIZE \
--loss_genre "Logistic" \
--gpu $GPU --num_thread $NUM_THREADS --mix_cpu_gpu \
--regularization_coef $REGULARIZATION_COEF \
-adv \
--gamma $GAMMA # FOR TRANSLATIONAL MODELS

