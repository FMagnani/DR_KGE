import pandas as pd
import json

with open('records.json', 'r') as f:
    records = json.load(f)

df = pd.DataFrame()

total_exp = records['total_exp']

model_list=[]
dataset_list=[]
max_step_list=[]
batch_size_list=[]
lr_list=[]
hidden_dim_list=[]
hits10_list=[]
hits50_list=[]
hits100_list=[]
for i in range(total_exp+1)[1:]:
    idx = str(i)
    exp = records[idx]
    model_list.append(exp['model'])
    dataset_list.append(exp['experiment_name'])
    max_step_list.append(exp['max_step'])
    batch_size_list.append(exp['batch_size'])
    lr_list.append(exp['learning_rate'])
    hidden_dim_list.append(exp['hidden_dim'])
    hits10_list.append(exp['Hits@10'])
    hits50_list.append(exp['Hits@50'])
    hits100_list.append(exp['Hits@100'])

df["Model"] = pd.Series(model_list)
df["Dataset"] = pd.Series(dataset_list)
df["Train Epochs"] = pd.Series(max_step_list)
df["Batch Size"] = pd.Series(batch_size_list)
df["Learning Rate"] = pd.Series(lr_list)
df["Embedding Dim"] = pd.Series(hidden_dim_list)
df["Hits@10"] = pd.Series(hits10_list)
df["Hits@50"] = pd.Series(hits50_list)
df["Hits@100"] = pd.Series(hits100_list)

df = df.sort_values(by=["Hits@10","Hits@50","Hits@100"], axis=0, ascending=[False,False,False])

print(df)

winner_name = str(df["Model"][0])
winner_idx = str(df.reset_index()["index"][0])
dataset = df["Dataset"][0]
winner_path = dataset+"/"+winner_name+"_"+dataset+"_"+winner_idx+"/"
winner_hits100 = pd.read_csv(winner_path+"HitsAt100.csv", index_col=0)
print("\nBest model's predictions:")
print(winner_hits100)
