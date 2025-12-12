
import numpy as np
import pandas as pd

from utilities import pull_data, get_X_features_crg, get_X_features_op

class Model:

    def __init__(self, crg_team):
        # features need to be hardcoded because they have unique preprocessing.
        self.crg_team = crg_team #options: VL or BS
        self.X_features = [
            'jam_id', 'lead', 'no_initial', 'trips', 'jammer_penalty_counter', 'blocker_penalty_counter',
            'op_jam_id', 'op_lead', 'op_no_initial', 'op_trips', 'op_jammer_penalty_counter', 'op_blocker_penalty_counter']
        self.X_skater_labels = ['jammer', 'blocker', 'pivot']
        self.X_columns = []
        self.y_label = 'point_diff'
        self.df_data = pull_data(self.crg_team)
        self.df_features = pd.DataFrame()

    def preprocess_data(self):
        print('hello')
        
        # TODO: Error checking
        self.df_features['point_diff'] = self.df_data['Jam Total'] - self.df_data['OP_Jam Total']

        self.df_features = get_X_features_crg(self.df_data, self.df_features, self.X_features)
        self.df_features = get_X_features_op(self.df_data, self.df_features, self.X_features)

    #def train(self):

    #def test(self):

    #def predict(self):

model = Model('VL')
model.preprocess_data()