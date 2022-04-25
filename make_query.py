import subprocess
import json

with open('config.json', 'r') as f:
    config = json.load(f)


    
EXPERIMENT_NAME = config["global"]["ExperimentName"]
MODEL_NAME = config["make_embedding"]["ModelName"]
MODEL_NUMBER = config["make_query"]["ModelNumber"]
K = config["make_query"]["K"]


MODEL_PATH = EXPERIMENT_NAME+"/"+MODEL_NAME+"_"+EXPERIMENT_NAME+"_"+str(MODEL_NUMBER)

line_1 = "dglke_predict --model_path \""+MODEL_PATH+"\" --format 'h_r_t' "
line_2 = "--data_files \""+EXPERIMENT_NAME+"/query_heads.csv\" \""+EXPERIMENT_NAME+"/query_rels.csv\" \""+EXPERIMENT_NAME+"/query_tails.csv\" "
line_3 = "--raw_data --entity_mfile \""+EXPERIMENT_NAME+"/entities.tsv\" --rel_mfile \""+EXPERIMENT_NAME+"/relations.tsv\" "
line_4 = "--score_func none --topK "+str(K)+" --output \""+MODEL_PATH+"/scores.tsv\""

command = line_1+line_2+line_3+line_4

subprocess.run(command, shell=True)

