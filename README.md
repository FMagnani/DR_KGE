# Drug Repurposing through Knowledge Graph Embedding

This repository reproduces the approach of https://github.com/gnn4dr/DRKG.

## Overview

### The dataset
The Drug Repurposing problem often is framed as a link prediction task on a biomedical knowledge graph (in this case the Drug Repurposing Knowledge Graph - DRKG). Many other graphs similar to this are available. Briefly, a knowledge graph is a heterogeneous multi-modal graph, in the sense that both the edges and the nodes of the graph are characterized by a label (i.e. they belong to different classes). For example, we here have nodes of type _Compound_, _Gene_, _disease_, etc and edges of type _Treats_, _Downregulates_, etc. 
The DRKG is made up of 97 K nodes and 5,8 M edges, the edges belong to 107 types of relation. The algorithm in practice only employs the information about the edge class, implicitly accounting for the corresponding node types.  

### The prediction
In order to identify the drugs that could treat the Covid-19 disease, we predict the presence of a relation of kind "_Treats_" (symbol "T") or "_Compound treats Disease_" (symbol "CtD") from all the compounds to each of the nodes related to a Covid disease (the list of target nodes taken into consideration is the file `query_tails.csv` in folder `drkg`, and it's taken from the repo https://github.com/gnn4dr/DRKG). Since the algorithm assigns a score to these predicitons, we are able to rank the most promising compounds: in this way we prioritize the drugs with respect to a target disease. Using the external validation list of compounds `external_validation.csv` we can compute the metrics Hits@1, Hits@50 and Hits@100.  

### The metric
The choice of the metric is related to the real-world application. Currently there are no hints, apart from the intuitions of physicians, on which drugs could be repurposed in order to treat new diseases.  
Machine Learning hopes to aid in this problem by restricting the focus onto a much smaller number of drugs, that are identified as possible candidates. Therefore, the final result is a set, and the most suitable metric is the Hits@K. The "hits" are often defined by the compounds currently under clinical trial (in this case too), but there are other options such as results of High Throughput Screening experiments.
Another popular metric is the AUROC curve, but in that case all the range of drugs is taken into consideration (and all of them contribute equally to the metric result), while in reality the focus is to prioritize a small number of them.  

### The query
The prediction step is called "query" in this repo, interpreting the link prediction as a 1-hop query. There are interesting works aiming at allowing multi-hops conjunctive queries, notably [this one](https://github.com/hyren/query2box). Such approaches would like to make more interpretable the prediction process. Related to this goal that are also many works that leverage Reinforcement Learning in order to find a meaningful path connecting the source and target nodes of a query.  
In this repo, however, to predict the result of a query correspond to predict the existence of a triplet (i.e. a link of a specific type between two nodes). This list of triplets is defined in the folder `drkg`: `query_heads`, `query_rels` and `query_tails` define respectively the source nodes, the relations and the target nodes. All the possible combinations of `(source, relation, target)` triplets are scored and ranked. That's not the only possibility, for other options see [the correspoding docs of dgl ke](https://aws-dglke.readthedocs.io/en/latest/predict.html). 

### The algorithm: Knowledge Graph Embedding
In few words, the algorithm embeds all the nodes and all the edge classes (not the edges, but their classes) into vectors and operations on such vectors, respectively. So, to each node is associated a vector, to each class of edges is associated a transformation (as a translation, a rotation, or an element-wise prodcut. In any case, all of these are parameterized, so to each class of edges is associated in practice a choice of these parameters). In this way, we go from the graph space into an Euclidean space, in which we dispose of a gradient that allows us to perform gradient descent. We train a global loss function based on the correct classification of the triplets as existing or not existing.  
For a list of the specific models belonging to this class of algorithms, see https://github.com/xinguoxia/KGE.

### Library used
It's employed the library [dgl-ke](https://github.com/awslabs/dgl-ke) built on the top of [dgl](https://github.com/dmlc/dgl). The main reason for this choice, instead of relying on [PyG](https://github.com/pyg-team/pytorch_geometric) or [pykeen](https://github.com/pykeen/pykeen) that are specific to this kind of algorithms (knowledge graph embeddings), is speed: dgl-ke allows distributed computation over many GPUs and CPUs at the same time. For quite large graphs, as nearly all the biomedical graphs used for drug repurposing, that's a neat advantage.  
**IMPORTANT** Actually the dgl-ke library gave me some problems. I couldn't manage to install the *latest*, but I couldn't employ the *stable* (a specific loss function was missing). I manually performed some changes, so the **correct version of dgl-ke to employ for this repo is this: https://github.com/FMagnani/dgl-ke**. Note that it works fine but some code relative to the built-in datasets has been removed.

### Some results  
| Model | Train Epochs** | Batch Size** | Learning Rate | Embedding Dim | Hits@10 | Hits@50 | Hits@100 |
|---|---|---|---|---|---|---|---|
| TransE L2 | 50 K | 1600 | 0.25 | 400 | 1 | 3 | 4 |
| TransE L1 | 50 K | 1600 | 0.25 | 400 | 1 | 3 | 3 |
| DistMult | 50 K | 1600 | 0.25 | 400 | 1 | 2 | 3 |
| ComplEx | 100 K | 1600 | 0.25 | 400 | 1 | 2 | 2 |  
| All | - | 1600 | 0.25 | 400 | 1 | 4 | 5 |  
  
Single models predictions:  
**TransE L2**  
| Compound Name | Rank | Target | Relation | 
|---|---|---|---|
| Ribavirin | 4 | MESH:D045169 | GNBR:Treats |
| Dexamethasone | 23 | MESH:D045169 | GNBR:Treats |
| Methylprednisolone | 41 | MESH:D045169 | GNBR:Treats |
| Oseltamivir | 82 | MESH:D045169 | GNBR:Treats |
  
**TransE L1**  
| Compound Name | Rank | Target | Relation | 
|---|---|---|---|
| Ribavirin | 7 | MESH:D045169 | GNBR:Treats |
| Methylprednisolone | 18 | MESH:D045169 | GNBR:Treats |
| Dexamethasone | 38 | MESH:D045169 | GNBR:Treats |
  
**DistMult**  
| Compound Name | Rank | Target | Relation | 
|---|---|---|---|
| Ribavirin | 3 | MESH:D045169 | GNBR:Treats |
| Dexamethasone | 13 | SARS-CoV2 orf3a | Hetionet:Treats |
| Methylprednisolone | 76 | SARS-CoV2 orf3a | Hetionet:Treats |
  
**ComplEx**  
| Compound Name | Rank | Target | Relation | 
|---|---|---|---|
| Ribavirin | 1 | MESH:D045169 | GNBR:Treats |
| Chloroquine | 44 | SARS-CoV2 M | GNBR:Treats |
  
** Train Epochs refers to the epochs made on a single batch on each computational unit, Batch Size refers to the batch of data loaded on each computational unit. The epochs made in total are (Train Epochs) x (Number of Processes = 8 in my setting).  
  
## Instructions - How to use the repo
The workflow of the program is as follows:

![alt text](https://github.com/FMagnani/DR_KGE/blob/master/images/DGLKE_Workflow.png)

### Needed files
It's needed to have defined the dataset and the query in a specific directory: `drkg`.  
The graph is represented in an **edgelist** format, i.e. a list of triplets `(source, relation, target)` in which the "sources" and the "targets" are unique identifiers of the nodes, while "relations" are unique identifiers of the class of the edges. In such a way a graph is defined through the list of all its directed and labeled edges. The list of all the nodes and edge's classes present in the graph will be worked out by the algorithm and saved into two files called `entities.tsv` and `relations.tsv`.  
The dataset must be given into the three files `train.txt`, `test.txt` and `valid.txt`, that are edgelists in tsv format (I made their place-holders as a reference). Due to the fact that an external validation is employed, the `valid.txt` file does **not** contain the validation triplets used for the metric. The real validation is external to the dataset.  
The query, i.e. the triplets to score, must be given in the three files `query_heads`, `query_rels` and `query_tails` in `drkg`. More info in the [dgl ke docs](https://dglke.dgl.ai/doc/predict.html).  

### Execution
All the code is to be executed from command line. From the graph to the final metric it's only 3 commands:  
1. `bash make_embedding` (configure the variables into the script itself)  
2. `bash make_query` (In this case you have to select the folder in which the embedding are stored)  
3. `python hits.py --model <model name> --folder <folder number>`, e.g. `python hits.py --model TransE L2 --folder 0`  
  
## Personal opinion
Graphs are cool. Large graphs are also cool. Anyway, to make embeddings of large graphs is not a cool activity, it quite sucks.  
Moreover, the Ribavirin, that is always ranked first, is actually present in the training set. So, it's not a real prediction. Leaving that out, these algorithms can predict basically two compunds, that I guess are the most obvious choices, that could be made even without this method. Personally, I also believe that the other ones, the Choloroquine and the Oseltamivir, are there totally by chance.  
This method sucks applied to this problem. Nonetheless... maybe I could try it on some kind of Pokèmon knowledge graph...
