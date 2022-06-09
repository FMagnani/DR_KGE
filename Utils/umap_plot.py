import numpy as np
import matplotlib.pyplot as plt
import umap
import umap.plot
import pandas as pd


def main():

	sample_size = 0.1 # Portion of embedding of each type to plot (they're sampled)

	emb_path = "/home/federico/Projects/link_prediction_DGLKE/drkg/TransE_l2_drkg_8/drkg_TransE_l2_entity.npy" # Path to the embedding	
	ent_path = "/home/federico/Projects/link_prediction_DGLKE/drkg/entities.tsv" # Path to the entities.tsv file

	### Load embeddings and entities list
	np_emb = np.load(emb_path)
	df_ent = pd.read_csv(ent_path, sep='\t', index_col=0, header=None, names=["Id"])
	
	types = df_ent["Id"].str.split(pat="::", expand=True)[0].values
	df_ent["Type"] = types
	
	types_to_plot = [
		"Compound",
		"Disease",
		"Atc",
		"Gene",
		"Pharmacologic Class",
		"Side Effect"
	]
	
	# Filter types - COMMENT THE LINE FOR NOT FILTERING
	df_ent = df_ent[ df_ent["Type"].isin(types_to_plot) ]

	# Sample embeddings
	df_ent = df_ent.groupby("Type").sample(frac=sample_size, random_state=1) # Sample index from df		
	index = df_ent.index
	np_emb = np_emb[index][:] # Take only the sampled entries from the embeddings
	
	print("Going to reduce "+str(np_emb.shape[0])+" points...\n")
	
	# Dimensionality reduction
	mapper = umap.UMAP(
		n_neighbors=15,
		random_state=74,
		n_components=2,
	).fit(np_emb)

	# Select width and height of the figure
	# If it's small and too many points overlap, datshading is automatically applied
	# For example, with all the entities of DRKG:
	# width, height = 1111, 900 NOT datashade
	# width, height = 800, 800  YES datashade
	width, height = 1111, 900

	figure = umap.plot.points(mapper, labels=df_ent["Type"], width=width, height=height, background='black')		
	
	plt.title(
		"UMAP visualization of embedding of DRKG",
		fontsize=16
	)
			
	plt.show()
	
	
if __name__=='__main__':
	
	main()	
