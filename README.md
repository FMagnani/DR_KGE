# Drug Repurposing through Knowledge Graph Embedding

This repository reproduces the approach of https://github.com/gnn4dr/DRKG. In that repo there is useful material and details about the construction of the dataset and some statistical analyses performed to it.  

## Overview

### The dataset
Conceptually, the Drug Repurposing problem is framed as a link prediction task on a biomedical knowledge graph, that in this case is the Drug Repurposing Knowledge Graph (DRKG). Many other graphs similar to this are available. Briefly, a knowledge graph is a heterogeneous directed graph, in the sense that the edges of the graph are characterized by a label (we can also say that they belong to a class).  
The DRKG is made up of 97,055 nodes and 5,869,294 edges, the edges belong to 107 types of relation. In principle, also the nodes are characterized by a type (as Compound, Disease, Gene, etc) but the algorithm in practice only employs the information about the edge class, implicitly accounting for the corresponding nodes.  

### The prediction
In order to identify the drugs that could treat the Covid-19 disease, we predict the presence of a relation of kind "treats" (symbol "T") or "Compound treats Disease" (symbol "CtD") from all the compounds to each of the nodes related to a Covid disease (the list of target nodes taken into consideration is the file `query_tails.csv` in folder `drkg`). Since the algorithm assigns a score to these predicitons, we are able to rank the compounds and subsequently compute the metrics Hits@1, Hits@50 and Hits@100.  

### The metric
The choice of the metric is related to the real-world application of such a procedure. Currently there are no hints, apart from the intuitions of physicians, on which already approved drugs could be repurposed in order to treat new diseases (or in any case different diseases than the ones they are currently used for). In practice, the community have to screen all of them.  
Machine Learning hopes to aid in this problem by restricting the focus onto a way smaller number of drugs, that are identified as possible candidates. Therefore, it's important this final set of identified compounds that will be taken into account by the medical community: the corresponding metric is the Hits@K.  
Another popular metric is the AUROC curve, but in that case all the range of drugs is taken into consideration (and all of them contribute equally to the metric result), while in reality the focus is to prioritize a small number of them.  

### The query
The prediction step is called "query" in this repo, interpreting the link prediction as a 1-hop query. There are interesting works aiming at allowing multi-hops conjunctive queries, notably [this one](https://github.com/hyren/query2box). Such approaches would like to make more interpretable the prediction process. Related to this goal that are also many works that leverage Reinforcement Learning in order to find a meaningful path connecting the source and target nodes of a query.  
In this repo many queries are made, scored and finally ranked. They are defined in the folder `drkg`:  
`query_heads`, `query_rels` and `query_tails` define respectively the source nodes, the relations to predict and the target nodes. All the possible combinations of `(source, relation, target)` triplets are scored and ranked. The rank, at the end, is interpreted as applying to the compound (the source node).  

### The algorithm
In few words, the algorithm embeds all the nodes and all the edge classes (not the edges, but their classes!). To each node is associated a vector in a vector space, to each class of edges is associated a transformation (as a translation, represented by a vector, or a rotation, represented by a matrix, etc) defined in the vector space in which the nodes are embedded.  
For a list of the specific models belonging to this class of algorithms, see https://github.com/xinguoxia/KGE.

### Library used
It's employed the library [dgl-ke](https://github.com/awslabs/dgl-ke) built on the top of [dgl](https://github.com/dmlc/dgl). The main reason for this choice, instead of relying on [PyG](https://github.com/pyg-team/pytorch_geometric) or [pykeen](https://github.com/pykeen/pykeen) that is specific to this kind of knowledge graph embeddings, is speed: dgl-ke allows distributed computation over a GPU and many CPU at the same time. For quite large graphs, as nearly all the biomedical graphs used for drug repurposing, that's a neat advantage.  
**IMPORTANT** Actually the dgl-ke library gave me some problems. I couldn't manage to install the *latest*, but I couldn't employ the *stable* (a specific loss function was missing). I manually performed some changes, so the **correct version of dgl-ke to employ for this repo is this: https://github.com/FMagnani/dgl-ke**. Note that it works fine but some code relative to the built-in datasets has been removed.

### Some results obtained  
| Model | Train Epochs** | Batch Size** | Learning Rate | Embedding Dim | Hits@10 | Hits@50 | Hits@100 |
|---|---|---|---|---|---|---|---|
| TransE l2 | 50000 | 1600 | 0.25 | 400 | 1 | 4 | 5 |
| TransE l2 | 100000 | 1600 | 0.25 | 400 | 1 | 4 | 5 |
| DistMult | 50000 | 1600 | 0.25 | 400 | 1 | 3 | 4 |
| TransE l1 | 50000 | 1000 | 0.25 | 400 | 1 | 3 | 4 |
| TransE l2 | 50000 | 3200 | 0.25 | 400 | 1 | 4 | 5 |
| ComplEx | 50000 | 1600 | 0.25 | 400 | 1 | 3 | 3 |  
  
Best model's predictions:
| Compound Name | Rank | Score*** | Relation 
|---|---|---|---|
| Ribavirin | 3 | -10.33 | Disease::MESH:D045169 | T |
| Dexamethasone | 11 | -12.59 | Disease::MESH:D045169 | T |
| Colchicine | 25 | -13.03 | Disease::MESH:D045169 | T |
| Methylprednisolone | 35 | -13.12 | Disease::MESH:D045169 | T |
| Deferoxamine | 71 | -13.42 | Disease::MESH:D045169 | T |
  
** Train Epochs refers to the Epochs made on each core, Batch Size refers to the batch of data loaded on each core.  
   The epochs made in total are (Train Epochs) x (Number of Processes = 8 in my setting)  
  
*** Score must be close to 0, it's negative since a margin = 10 has been employed   

## Instructions - How to use the repo
The workflow of the program is as follows:

![alt text](https://github.com/FMagnani/DR_KGE/blob/master/images/Workflow.png)

### Input files
It's needed to have defined the dataset and the query in a specific directory, as `drkg`.  
The graph is represented in an **edgelist** format, i.e. a list of triplets `(source, relation, target)` in which the "sources" and the "targets" are unique identifiers of the nodes, while "relations" are unique identifiers of the class of the edges. In such a way a graph is defined through the list of all its directed and labeled edges. The list of all the nodes and edge's classes present in the graph will be worked out by the algorithm and saved into two files called `entities.tsv` and `relations.tsv`.  
The dataset must be given into the three files `train.txt`, `test.txt` and `valid.txt`. that are edgelists in tsv format. Due to the fact that an external validation is employed, the `valid.txt` file does **not** contain the validation triplets used for the metric. The metric and validation is all done through the query.  
The query, that in fact constitues the validation, must be given as three files that list the nodes or relations to combine in all possible ways in order to build the triplets used for the link prediction.  

### Config file
When make_embedding is run, a new folder in the experiment folder is created. For example, inside `drkg` is created a folder `TransE_l2_drkg_0` meaning that you used the TransE model, over the drkg graph, and that's the first time. If you run it again with TransE, maybe modifying the embedding dimensions, that will be stored in a new folder called `TransE_l2_drkg_1`.  
In `config.json` are stored the most important hyperparameters:  

| | |
|---|---|
| ExperimentName | As "drkg". Name of the directory into which the graph and the query are stored |
| make_embedding | Details of all these hyperpars [here](https://aws-dglke.readthedocs.io/en/latest/train.html) |
| make_query, ModelNumber | The number of the folder contatining the embedding over which you want to make the query. For example, for `TransE_l2_drkg_1` the ModelNumber is 1 |
| make_query, K | The top K scored triplets to save into results. It's set to 110 since we use at most Hits@100 |

### Automatic Pipeline
Having set `config.json` you can ran the whole pipeline with `main.py`. It simply compiles all the scripts and exectues them.  
Alternatively, you can run them one by one.  
The files `make_embedding.py` and `make_query.py` simply read the configuration and execute `make_embedding.sh` and `make_query.sh` respectively. The shell scripts are given as a reference. For other examples see [this](https://github.com/awslabs/dgl-ke/blob/master/notebook-examples/kge_wikimedia.ipynb).  
The script `compute_hits.py` computes the metric and saves the results into `records.json`, that can be inspected using `inspect_records.py`. The mapping from the compound identifiers and their common names is coded inside the script.  



