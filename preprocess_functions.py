import pandas as pd
import numpy as np

def pull_data(crg_team):

    # TODO: Error checking

    filepath = "C:/Users/trfit/OneDrive/Documents/Hobbies/CRG/stats 2025/prepped_data/" 
    if crg_team == 'VL':
        files = [
            "VL_Akron_data.csv",
            "VL_Athens_data.csv",
            "VL_Terrorz_data.csv"]
    elif crg_team == 'BS':
        BS_files = [
            "BS_Akron_data.csv",
            "BS_Athens_data.csv",
            "BS_Great_Lakes_data.csv",
            "BS_Louisville_data.csv"] 
    
    df_lst = []
    for file in files:
        df_lst.append(pd.read_csv(filepath+file))
    return pd.concat(df_lst, ignore_index=True)

def build_encoded_matrix(df, cols, skaters):
    # This function calculates the average normalized jam +/- for each skater using matrix multiplication. This method is used because it makes 
    # it easier to remove a skater and recalculate.

    df_encoded = pd.get_dummies(df, columns=cols)

    # Combine the position boolean columns, so each skater has one boolean column, True = skater was in the jam, False = skater was not in the jam
    #cols = ['Jammer', 'Pivot', 'Blocker 1', 'Blocker 2', 'Blocker 3']
    for skater in skaters:
        cols_comb = []
        for col in cols:
            if col+"_"+str(skater) in df_encoded.columns:
                cols_comb.append(col+"_"+str(skater))
        df_encoded[str(skater)] = df_encoded[cols_comb].any(axis='columns')
    
        for col in cols_comb:
            df_encoded = df_encoded.drop(col, axis=1)

    return df_encoded


def get_X_features_crg(df_data, df_features, feature_list):

    df_op = pd.get_dummies(df_data['OP'], columns = 'OP')

    for col in df_op:
        df_features[col] = df_op[col].astype(int)
        feature_list.append(col)

    if 'jam_id' in feature_list:
        for col in ['OP', 'Half',  'Jam']:
            df_data[col] = df_data[col].astype("string")
        df_features['jam_id'] = df_data['OP'] +"_"+ df_data['Half'] +"_"+ df_data['Jam']
    
    if 'lead' in feature_list:
        df_features['lead'] = df_data['LEAD'].replace({np.nan: False, "X": True}).infer_objects(copy=False)
    
    if 'no_initial' in feature_list:
        df_features['no_initial'] = df_data['NI'].replace({np.nan: False, "X": True}).infer_objects(copy=False)
    
    if 'trips' in feature_list:
        df_features['trips'] = df_data['Trips']

    if 'jammer_penalty_counter' in feature_list:
        for col in ['Jammer_Box_1', 'Jammer_Box_2', 'Jammer_Box_3']:
            df_data[col] = df_data[col].replace({np.nan: 0, '+': 1, '-': 1, '$': 1}).infer_objects(copy=False)
        df_features['jammer_penalty_counter'] = df_data['Jammer_Box_1'] + df_data['Jammer_Box_2'] + df_data['Jammer_Box_3']

    if 'blocker_penalty_counter' in feature_list:
        df_features['blocker_penalty_counter'] = 0
        for col in ['Pivot_Box_1','Pivot_Box_2','Pivot_Box_3',
                    'Blocker_1_Box_1','Blocker_1_Box_2','Blocker_1_Box_3',
                    'Blocker_2_Box_1','Blocker_2_Box_2','Blocker_2_Box_3',
                    'Blocker_3_Box_1','Blocker_3_Box_2','Blocker_3_Box_3']:

            df_data[col] = df_data[col].replace({np.nan: 0, '+': 1, '-': 1, '$': 1, 'S': 1, '3':0}).infer_objects(copy=False)
            df_features['blocker_penalty_counter'] += df_data[col]
    
    #return df_features

def get_X_features_op(df_data, df_features, feature_list):
    
    if 'op_lead' in feature_list:
        df_features['op_lead'] = df_data['OP_LEAD'].replace({np.nan: False, "X": True}).infer_objects(copy=False)
    
    if 'op_no_initial' in feature_list:
        df_features['op_no_initial'] = df_data['OP_NI'].replace({np.nan: False, "X": True}).infer_objects(copy=False)
    
    if 'op_trips' in feature_list:
        df_features['op_trips'] = df_data['OP_Trips']

    if 'op_jammer_penalty_counter' in feature_list:
        for col in ['OP_Jammer_Box_1', 'OP_Jammer_Box_2', 'OP_Jammer_Box_3']:
            df_data[col] = df_data[col].replace({np.nan: 0, '+': 1, '-': 1, '$': 1}).infer_objects(copy=False)
        df_features['op_jammer_penalty_counter'] = df_data['OP_Jammer_Box_1'] + df_data['OP_Jammer_Box_2'] + df_data['OP_Jammer_Box_3']

    if 'op_blocker_penalty_counter' in feature_list:
        df_features['op_blocker_penalty_counter'] = 0
        for col in ['Pivot_Box_1','Pivot_Box_2','Pivot_Box_3',
                    'Blocker_1_Box_1','Blocker_1_Box_2','Blocker_1_Box_3',
                    'Blocker_2_Box_1','Blocker_2_Box_2','Blocker_2_Box_3',
                    'Blocker_3_Box_1','Blocker_3_Box_2','Blocker_3_Box_3']:

            df_data['OP_'+col] = df_data['OP_'+col].replace({np.nan: 0, '+': 1, '-': 1, '$': 1, 'S': 1,'s': 1, '3':0}).infer_objects(copy=False)
            df_features['op_blocker_penalty_counter'] += df_data['OP_'+col]
    
    #return df_features

def get_X_skater_labels(df_data, df_features, feature_list, pivot=True):

    blocker_cols = ['Blocker_1', 'Blocker_2', 'Blocker_3']
    if pivot:
        df_concated = pd.concat([df_data['Blocker_1'],df_data['Blocker_2'],df_data['Blocker_3']]).unique()
        
        df_pivot= pd.get_dummies(df_data['Pivot'], columns = 'Pivot')
        for col in df_pivot:
            df_features[str(col)+"_pivot"] = df_pivot[col].astype(int)
            feature_list.append(str(col)+"_pivot")
    
    else:
        blocker_cols.append('Pivot')
        df_concated = pd.concat([df_data['Pivot'],df_data['Blocker_1'],df_data['Blocker_2'],df_data['Blocker_3']]).unique()

    
    df_encoded = build_encoded_matrix(df_data[blocker_cols], blocker_cols, df_concated) 
    feature_list += [col for col in df_encoded.columns]
    df_features = pd.concat([df_features, df_encoded], axis=1)

    
    df_jammer= pd.get_dummies(df_data['Jammer'], columns = 'Jammer')
    for col in df_jammer:
        df_features[str(col)+"_jammer"] = df_jammer[col].astype(int)
        feature_list.append(str(col)+"_jammer")
    
    # Note: why is return needed here? why isn't this working quite right?
    return df_features




        