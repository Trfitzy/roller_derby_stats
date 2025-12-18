import numpy as np
import pandas as pd
from model import Model
team = 'BS'
model = Model(team)
model.preprocess_data()

#model.train_test_known_data()

model.train(model.df_features)

# Test 1: blockers (pivot included)
df_pull = pd.read_csv(team+"_jammer_test.csv")

df = df_pull[df_pull['length'] >= 4].reset_index(drop=True)
#print(df)

# Create test dataframe
df_test = pd.DataFrame([[0]*len(model.X_columns)]*len(df), columns=model.X_columns)
df_test[model.y_label] = df['avg_point_diff']
i = -1
for pack in df['itemsets']:
    i += 1
    pack = pack.removeprefix("frozenset({").removesuffix("})").replace("'","").replace(" ","")
    pack=pack.split(",")
    for skater in pack:#list(pack):
        df_test.loc[[i],[skater]] = 1

for col in model.X_features:
    df_test[col] = model.df_features[col].mean()

model.test(df_test)