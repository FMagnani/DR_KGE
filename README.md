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
More details are given in the dedicated paragraph.  
For a list of the specific models belonging to this class of algorithms, see https://github.com/xinguoxia/KGE.

### Library used
It's employed the library [dgl-ke](https://github.com/awslabs/dgl-ke) built on the top of [dgl](https://github.com/dmlc/dgl). The main reason for this choice, instead of relying on [PyG](https://github.com/pyg-team/pytorch_geometric) or [pykeen](https://github.com/pykeen/pykeen) that is specific to this kind of knowledge graph embeddings, is speed: dgl-ke allows distributed computation over a GPU and many CPU at the same time. For quite large graphs, as nearly all the biomedical graphs used for drug repurposing, that's a neat advantage.

## Instructions - How to use the repo
The workflow of the program is as follows:

![alt text](https:...)

### Make embedding and predictions


