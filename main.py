"""
You should always have the following files in the dataset's directory:
- train.txt, test.txt, valid.txt

Moreover, for specific link prediction tasks (ex. drug repurposing), you should have:
- query_heads.csv, query_tails.csv, query_rels.csv
in order to perform a meaningful validation.

For the benchmark dataset FB15k-237, the validation is general and it is done automatically using --test in dglke_train.

Remember to set the conifguration file config.json!
"""
import json

with open('config.json', 'r') as f:
	config = json.load(f)

ExpName = config["global"]["ExperimentName"]

with open("make_embedding.py") as f:
	        make_embedding_code = compile(f.read(), "make_embedding.py", 'exec')

if ExpName=="drkg": 
        
	with open("make_query.py") as f:
	        make_query_code = compile(f.read(), "make_query.py", 'exec')

	with open("compute_hits.py") as f:
	        compute_hits_code = compile(f.read(), "compute_hits.py", 'exec')

	with open("inspect_records.py") as f:
	        inspect_records_code = compile(f.read(), "inspect_records.py", 'exec')


	print("Going to make embedding\n")
	exec(make_embedding_code)

	print("Embedding done!\nGoing to perform the query\n")
	exec(make_query_code)

	print("Query performed!\nGoing to compute and save Hits@K metrics.")
	exec(compute_hits_code)

	print("Metric saved!\nHere all the results about this dataset.")
	exec(inspect_records_code)


else:

	print("Going to make the embedding and test it. Metric results will printed in terminal.\n")
	exec(make_embedding_code)
	
