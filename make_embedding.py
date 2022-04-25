import subprocess
import json

with open('config.json', 'r') as f:
    config = json.load(f)

EXPERIMENT_NAME = config["global"]["ExperimentName"]
FORMAT = config["make_embedding"]["Format"]
# MODEL
MODEL_NAME = config["make_embedding"]["ModelName"]
GAMMA = config["make_embedding"]["Gamma"]  # For translational models
LR = config["make_embedding"]["LR"]
HIDDEN_DIM = config["make_embedding"]["HiddenDim"]
# TRAINING
MAX_STEP = config["make_embedding"]["MaxStep"]
BATCH_SIZE = config["make_embedding"]["BatchSize"]
NEG_SAMPLE_SIZE = config["make_embedding"]["NegSampleSize"]
LOG_INTERVAL = config["make_embedding"]["LogInterval"]
###
BATCH_SIZE_EVAL = config["make_embedding"]["BatchSizeEval"]
NEG_SAMPLE_SIZE_EVAL = config["make_embedding"]["NegSampleSizeEval"]
REGULARIZATION_COEF = config["make_embedding"]["RegularizationCoef"]
# HARDWARE
GPU = config["make_embedding"]["GPU"]
NUM_THREADS = config["make_embedding"]["NumThreads"]

bash_command = "dglke_train --dataset "+EXPERIMENT_NAME 

line_1 = "dglke_train --dataset \""+EXPERIMENT_NAME+"\" --data_path \""+EXPERIMENT_NAME+"/\" --save_path \""+EXPERIMENT_NAME+"/\" "
line_2 = "--model_name \""+MODEL_NAME+"\" --data_files \"train.txt\" \"valid.txt\" \"test.txt\" --format \""+FORMAT+"\" "
line_3 = "--lr "+str(LR)+" --max_step "+str(MAX_STEP)+" --log_interval "+str(LOG_INTERVAL)+" "
line_4 = "--hidden_dim "+str(HIDDEN_DIM)+" "
line_5 = "--batch_size "+str(BATCH_SIZE)+" --neg_sample_size "+str(NEG_SAMPLE_SIZE)+" "
line_6 = "--loss_genre \"Logistic\" "
line_7 = "--gpu "+str(GPU)+" --num_thread "+str(NUM_THREADS)+" --mix_cpu_gpu "
line_8 = "--regularization_coef "+str(REGULARIZATION_COEF)+" -adv --gamma "+str(GAMMA)

command = line_1 + line_2 + line_3 + line_4 + line_5 + line_6 + line_7 + line_8

subprocess.run(command, shell=True)

