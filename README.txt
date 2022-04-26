#############################
##   Pipeline 1-hop-query  ##
##    From DRKG paper      ##
#############################

## DATA
The initial data must be in edgelist format and csv.
It should have head, relations and tail columns with names HeadId,RelType,TailId. 
All will be treated as Id's. 
The node's type doesn't matter (it's defined by the different relation types). Only the relation type does.

## INSTRUCTIONS

# ADD A NEW DATASET
1- Make the folder of the project, with a new name
2- Put the initial dataset into the folder in csv format, with name NAME_data.csv
3- Run "python prepare_data.py", to split the dataset into train, test, validation
4- You should define the query, i.e. a file of source nodes, a file of relations, a file of tail nodes. 
 - You can use "python define_query.py" (it should be modified according to your needs) or make the files in your own way.

# MAKE EMBEDDING AND PREDICTIONS
Assuming that all the needed files are present, i.e. train.txt, test.txt, valid.txt, query_heads.csv, query_rels.csv, query_tails.csv:

1- "conda activate KG_EMB", to activate the environment
2- Specify the configuration in:
     - config.json          <-- General config
     - make_embedding.sh    <-- Embedding config
     - make_query.sh        <-- Prediction config  
3- "bash make_embedding.sh", to realize the embedding
4- "bash make_query.sh", to make the query and get results
5- "python compute_hits.py", to save the results of Hits@K into "record". 
6- "python inspect_records.py"
7- "conda deactivate" when finished the session

## OUTPUT
At the end, in the experiment folder, you will have:
- original data NAME_data.csv
- test.txt, train.txt, valid.txt
- entities.tsv, relations.tsv (the lists of IDs)
- query_heads.csv, query_tails.csv, query_rels.csv
- results.csv (of the query: ranking and score) 
A number of folders with single embeddings given by different configurations, in each:
-- config.json (config used)
-- NAME_entity.npy (entities embedding)
-- NAME_relation.npy (relation embedding)

