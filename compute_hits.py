import pandas as pd
import json
#import matplotlib.pyplot as plt
import numpy as np
#import scikitplot as skplt

EXPERIMENT_NAME="drkg"
MODEL_NAME="TransE_l2"
MODEL_NUMBER="6"

# Check if this result has already been saved
with open('records.json', 'r') as f:
    records = json.load(f)

total_exp = records['total_exp']

for i in range(total_exp+1)[1:]:
    exp = records[str(i)]
    name_check = (exp['experiment_name']==EXPERIMENT_NAME)
    model_check = (exp['model']==MODEL_NAME)
    number_check = (exp['no.']==MODEL_NUMBER)
    if (name_check and model_check and number_check):
        raise ValueError('This results are already saved in "records"!')


PATH = EXPERIMENT_NAME+"/"+MODEL_NAME+"_"+EXPERIMENT_NAME+"_"+MODEL_NUMBER+"/"

# Load results as df in a form like
#  Rank     CompoundId     score
#   1         DB0057       -8.2
#   2         DB5286       -9.3
#  ...          ...         ...
scores = pd.read_csv(PATH+"scores.tsv", sep='\t')
scores["CompoundId"] = scores["head"].str.rsplit(pat=":", n=1, expand=True)[1]
scores = scores.reset_index()
scores = scores.rename(columns={"index":"Rank"})

validation_data_id = [
    "DB00746",
    "DB05511",
    "DB00678",
    "DB01050",
    "DB12466",
    "DB08877",
    "DB01234",
    "DB01041",
    "DB00302",
    "DB06273",
    "DB11767",
    "DB12580",
    "DB11720",
    "DB00198",
    "DB11817",
    "DB00020",
    "DB00608",
    "DB00026",
    "DB12534",
    "DB00207",
    "DB14066",
    "DB00811",
    "DB08895",
    "DB09036",
    "DB09035",
    "DB00435",
    "DB01394",
    "DB14761",
    "DB01611",
    "DB01257",
    "DB00959"
]

validation_data_name = [
    "Deferoxamine",
    "Piclidenoson",
    "Losartan",
    "Ibuprofen",
    "Favipiravir",
    "Ruxolitinib",
    "Dexamethasone",
    "Thalidomide",
    "Tranexamic acid",
    "Tocilizumab",
    "Sarilumab",
    "Tradipitant",
    "Angiotensin_1-7",
    "Oseltamivir",
    "Baricitinib",
    "Sargramostim",
    "Chloroquine",
    "Anakinra",
    "Mavrilimumab",
    "Azithromycin",
    "Tetrandrine",
    "Ribavirin",
    "Tofacitinib",
    "Siltuximab",
    "Nivolumab",
    "Nitric Oxide",
    "Colchicine",
    "Remdesivir",
    "Hydroxychloroquine",
    "Eculizumab",
    "Methylprednisolone"
]

# Here you have
#    CompoundId     CompoundName
#     DB0056         Deferoasko
#     DB0036          Ciullok
#      ...              ...
validation_df = pd.DataFrame()
validation_df["CompoundId"] = pd.Series(validation_data_id)
validation_df["CompoundName"] = pd.Series(validation_data_name)

# You merge but you only retain the entries in common
# So you basically just filtered the initial "scores" df
results = pd.merge(
    validation_df, scores,
    how='inner',
    on='CompoundId',
    sort=False,
    copy=False
    )


results = results.sort_values(by="Rank", axis=0)
results.reset_index(inplace=True)
results = results[["CompoundName", "CompoundId", "Rank", "rel", "tail", "score"]]

results_hits10 = results[results["Rank"]<10]
results_hits50 = results[results["Rank"]<50]
results_hits100 = results[results["Rank"]<100]

results_hits100[["CompoundName", "Rank", "score", "tail", "rel"]].to_csv(PATH+"HitsAt100.csv")

hits_log = {
    "hits10": len(results_hits10.index),
    "hits50": len(results_hits50.index),
    "hits100": len(results_hits100.index)
}
print(hits_log)

# SAVE TO RECORDS

with open(PATH+'config.json', 'r') as f:
    config = json.load(f)

records['total_exp'] = total_exp+1

new_entry = {
    'experiment_name': EXPERIMENT_NAME,
    'model': MODEL_NAME,
    'no.': MODEL_NUMBER,
    'max_step': config['max_step'],
    'batch_size': config['batch_size'],
    'hidden_dim': config['hidden_dim'],
    'learning_rate': config['lr'],
    'Hits@10': hits_log['hits10'],
    'Hits@50': hits_log['hits50'],
    'Hits@100': hits_log['hits100']
}

records[str(total_exp+1)] = new_entry

with open('records.json', 'w', encoding='utf-8') as f:
    json.dump(records, f, ensure_ascii=False, indent=4)

"""
AUROC

results["labels"] = results["CompoundId"].isin(validation_data_id)

def compute_rates(df):

    thresholds = np.linspace(df["score"].min(),df["score"].max(), 30)

    tp_rate = []
    fp_rate = []
    for t in thresholds:
        tmp = df["score"]<t
        df["class"] = tmp.to_list()
        df = df[:][ df["class"] ]

        N = df.shape[0]
        if not (N==0):
            true_pos = df["class"]==df["labels"]
            tmp = true_pos.shape[0]/N
            tp_rate.append( tmp )
            fp_rate.append( 1-tmp )

    return fp_rate, tp_rate

fal_rate, pos_rate = compute_rates(results[["score","labels"]])

print(fal_rate)
print(pos_rate)

plt.plot(pos_rate, fal_rate, 'o')
plt.show()
"""
