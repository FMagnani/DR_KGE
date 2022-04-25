import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import json

"""
Script

IN:  EXPERIMENT_NAME_data.csv       from  EXPERIMENT_NAME/
OUT: train.txt, test.txt, valid.txt  in   EXPERIMENT_NAME/
     node_labels.csv                 in   EXPERIMENT_NAME/

folders: EXPERIMENT_NAME/ ---> EXPERIMENT_NAME/

files: EXPERIMENT_NAME_data.csv ---> train.txt, test.txt, valid.txt, node_labels.csv
"""

with open('config.json', 'r') as f:
    config = json.load(f)

EXPERIMENT_NAME = config['global']['ExperimentName']
TRAIN_SIZE = config['prepare_data']['TrainSize']
TEST_TO_VALID_SIZE = config['prepare_data']['TestToValidSizeRatio']
SEED_1 = config['prepare_data']['RandomSeed1']
SEED_2 = config['prepare_data']['RandomSeed2']

load_path = EXPERIMENT_NAME+"/"+EXPERIMENT_NAME+"_data.tsv"

df = pd.read_csv(load_path, sep='\t', header=None, names=['HeadId','RelType','TailId'])

# Prepare data for training
df_for_training = df[['HeadId','RelType','TailId']]

trainDF, testDF = train_test_split(df_for_training, train_size=TRAIN_SIZE, random_state=SEED_1)
testDF, validDF = train_test_split(testDF, test_size=TEST_TO_VALID_SIZE, random_state=SEED_2)

total_edges = df_for_training.shape[0]
train_edges = trainDF.shape[0]
test_edges = testDF.shape[0]
valid_edges = validDF.shape[0]

print("Total edges: ", total_edges,
      "\nTrain edges: ", train_edges, " (", train_edges/total_edges*100, "%)\n",
      "\nTest edges: ", test_edges, " (", test_edges/total_edges*100, "%)\n",
      "\nValid edges: ", valid_edges, " (", valid_edges/total_edges*100, "%)\n"
     )

# TRAINING TEST VALIDATION DATA
# In tsv format but txt extension
# Training needs the edge attributes but can't handle node types
# The format is called raw_uud_hrt, see https://aws-dglke.readthedocs.io/en/latest/train.html
save_path_training = EXPERIMENT_NAME+"/"

trainDF.to_csv(save_path_training+"train.txt", header = None, index = None, sep = "\t")
testDF.to_csv(save_path_training+"test.txt", header = None, index = None, sep = "\t")
validDF.to_csv(save_path_training+"valid.txt", header = None, index = None, sep = "\t")

