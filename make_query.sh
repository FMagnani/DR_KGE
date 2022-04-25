#!/bin/bash

# For reference see the following links:
# https://aws-dglke.readthedocs.io/en/latest/predict.html

EXPERIMENT_NAME="full_drkg"
MODEL_NAME="TransE_l2"
MODEL_NUMBER="6"
K=1000


MODEL_PATH=$EXPERIMENT_NAME$"/"$MODEL_NAME$"_"$EXPERIMENT_NAME$"_"$MODEL_NUMBER

dglke_predict --model_path $MODEL_PATH --format 'h_r_t' \
--data_files $EXPERIMENT_NAME$"/query_heads.csv" $EXPERIMENT_NAME$"/query_rels.csv" $EXPERIMENT_NAME$"/query_tails.csv" \
--raw_data --entity_mfile $EXPERIMENT_NAME$"/entities.tsv" --rel_mfile $EXPERIMENT_NAME$"/relations.tsv" \
--score_func none --topK $K --output $MODEL_PATH$"/scores.tsv"
