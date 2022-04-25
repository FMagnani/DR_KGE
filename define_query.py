import pandas as pd
import numpy as np
import json

"""
Script

Load the list of entity's identifiers and returns 3 files:
    query_heads: The compounds
    query_rels: The TREAT relation
    query_tails: The COVID-related disease entities

Modify this file in order to define different queries, as:
    query_heads: compound-related genes
    query_rels: gene-gene interaction
    query_tails: genes associated to COVID diseases
As it's done in DRKG, for example.
"""

with open('config.json', 'r') as f:
    config = json.load(f)

EXPERIMENT_NAME = config['global']['ExperimentName']

labels_raw = pd.read_csv(EXPERIMENT_NAME+"/entities.tsv",
                         sep='\t', index_col=0,
                         header=None, names=["Id"])

# The following 4 lines initialize "labels" as the list of entities + a column for their type
labels = labels_raw["Id"].str.split(pat="::", expand=True)[0]
labels.rename("Type", inplace=True)
labels = pd.DataFrame(labels)
labels["Id"] = labels_raw["Id"]

relations = pd.read_csv(EXPERIMENT_NAME+"/relations.tsv",
                        sep='\t', index_col=0,
                        header=None, names=["Id"])

# Change this to specify the set of target nodes
TargetSet = [
    "Disease::SARS-CoV2 E",
    "Disease::SARS-CoV2 N",
    "Disease::SARS-CoV2 nsp1",
    "Disease::SARS-CoV2 nsp12",
    "Disease::SARS-CoV2 nsp14",
    "Disease::SARS-CoV2 nsp2",
    "Disease::SARS-CoV2 nsp5",
    "Disease::SARS-CoV2 nsp5 C145A",
    "Disease::SARS-CoV2 nsp7",
    "Disease::SARS-CoV2 nsp9",
    "Disease::SARS-CoV2 orf3a",
    "Disease::SARS-CoV2 orf6",
    "Disease::SARS-CoV2 orf8",
    "Disease::MESH:D045169",
    "Disease::MESH:D001351",
    "Disease::MESH:D028941",
    "Disease::MESH:D006517",
    "Disease::SARS-CoV2 M",
    "Disease::SARS-CoV2 Spike",
    "Disease::SARS-CoV2 nsp10",
    "Disease::SARS-CoV2 nsp13",
    "Disease::SARS-CoV2 nsp15",
    "Disease::SARS-CoV2 nsp4",
    "Disease::SARS-CoV2 nsp11",
    "Disease::SARS-CoV2 nsp6",
    "Disease::SARS-CoV2 nsp8",
    "Disease::SARS-CoV2 orf10",
    "Disease::SARS-CoV2 orf3b",
    "Disease::SARS-CoV2 orf7a",
    "Disease::SARS-CoV2 orf9b",
    "Disease::MESH:D045473",
    "Disease::MESH:D065207",
    "Disease::MESH:D058957",
    "Disease::SARS-CoV2 orf9c"
    ]

# Change this to specify the set of relations to take into account
RelationSet = [
    "T",
    "CtD"
    ]

# Change this to specify the class to use as source nodes
query_heads = labels["Type"]=="Compound"

query_tails = labels["Id"].isin(TargetSet)
query_rels = relations["Id"].isin(RelationSet)

#labels.reset_index()["index"][query_heads].to_csv(EXPERIMENT_NAME+"/query_heads.csv",
#                                                    index=False, header=False)
#labels.reset_index()["index"][query_tails].to_csv(EXPERIMENT_NAME+"/query_tails.csv",
#                                                    index=False, header=False)
#relations.reset_index()["index"][query_rels].to_csv(EXPERIMENT_NAME+"/query_rels.csv",
#                                                    index=False, header=False)

# The following defines the query with so called RAW INDEX
# See https://aws-dglke.readthedocs.io/en/latest/predict.html for details
labels["Id"][query_heads].to_csv(EXPERIMENT_NAME+"/query_heads.csv",
                                  index=False, header=False)
labels["Id"][query_tails].to_csv(EXPERIMENT_NAME+"/query_tails.csv",
                                  index=False, header=False)
relations["Id"][query_rels].to_csv(EXPERIMENT_NAME+"/query_rels.csv",
                                    index=False, header=False)
