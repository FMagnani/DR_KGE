import pandas as pd
import argparse
import json

def main(args):
	
	# Load external validation list
	validation = pd.read_csv("external_validation.csv")
	
	# Load scores of triplets
	folder_path = "drkg/"+args.model+"_drkg_"+args.folder+"/"
	scores = pd.read_csv(folder_path+"scores.tsv", sep='\t')
	scores["CompoundId"] = scores["head"].str.rsplit(pat=":", n=1, expand=True)[1] # Extract the id of the compounds
	scores = scores.reset_index()
	scores = scores.rename(columns={"index":"Rank"}) # add a column with rank
	scores["Rank"] = scores["Rank"] +1 # Make the rank start at 1 (and not at 0)
		
	results = pd.merge(
		validation, 
		scores,
		how='inner',
		on='CompoundId',
		sort=False,
		copy=False
	)

	results = results.sort_values(by="Rank", axis=0) # Sort by rank
	results.reset_index(inplace=True) # When sorting, it keeps the old index. Here reset it.
	results = results[["CompoundId", "CompoundName", "Rank", "rel", "tail", "score"]]
	
	print("\nAll hits: \n", results)

	# Before to compute hits,
	# check for multiple occurrences of compounds (due to it being present in different triplets)
	unique_compounds = results["CompoundName"].unique()
	if not len(unique_compounds) == len(results.index):
	
		# The following basically only the unique compound names, with the highest rank
		to_keep_idx = []
		for compound in unique_compounds:
			filt_df = results[ results["CompoundName"]==compound ]
			best_scoring_idx = filt_df.iloc[0].name
			to_keep_idx.append(best_scoring_idx)
    
		results  = results[results.index.isin(to_keep_idx)]

	hits_at_10 = len(results[results["Rank"]<10].index)
	hits_at_50 = len(results[results["Rank"]<50].index)
	hits_at_100 = len(results[results["Rank"]<100].index)
	
	print("\nHits at 10:  "+str(hits_at_10)+"\nHits at 50:  "+str(hits_at_50)+"\nHits at 100: "+str(hits_at_100))

	with open(folder_path+'config.json', 'r') as f:
		config = json.load(f)

	print("\nConfiguration:")
	print("Max step:        "+str(config["max_step"]))
	print("Batch size:      "+str(config["batch_size"]))
	print("Neg sample size: "+str(config["neg_sample_size"]))
	print("Hidden dim:      "+str(config["hidden_dim"]))
	print("Learning rate:   "+str(config["lr"]))
	print("Loss:            "+str(config["loss_genre"])+'\n')
	
if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Compute hits from a ranked list of triplets')

	parser.add_argument("--model", type=str, default="TransE_l2", choices=['TransE_l2', 'TransE_l1', 'DistMult', 'ComplEx'],
						help="Name of the model")

	parser.add_argument("--folder", type=str, default='0',
						help="Number of the folder")

	args = parser.parse_args()
	main(args)
