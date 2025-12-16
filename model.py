
import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, KFold
import matplotlib.pyplot as plt

from preprocess_functions import pull_data, get_X_features_crg, get_X_features_op, get_X_skater_labels, plot_results

class Model:

    def __init__(self, crg_team, alpha=20):
        # features need to be hardcoded because they have unique preprocessing.
        self.crg_team = crg_team #options: VL or BS
        self.X_features = [
           #'jam_id', 
           'lead', 'no_initial', 'trips', 
            'jammer_penalty_counter', 'blocker_penalty_counter',
            'op_lead', 'op_no_initial', 'op_trips', 
            'op_jammer_penalty_counter', 'op_blocker_penalty_counter'
            ]
        
        self.X_skater_labels = []
        self.X_columns = []
        self.y_label = 'point_diff'
        self.alpha = alpha
        self.df_data = pull_data(self.crg_team)
        self.df_features = pd.DataFrame()
        self.model = Ridge(alpha=self.alpha)

        self.alpha=20

    def preprocess_data(self):
        
        # TODO: Error checking
        self.df_features['point_diff'] = self.df_data['Jam Total'] - self.df_data['OP_Jam Total']

        # TODO: Error checking for unexected characters
        get_X_features_crg(self.df_data, self.df_features, self.X_features)
        get_X_features_op(self.df_data, self.df_features, self.X_features)
        self.df_features = get_X_skater_labels(self.df_data, self.df_features, self.X_skater_labels, pivot=False)
    
        self.X_columns = self.X_features + self.X_skater_labels

    def train_test_known_data(self):
        ridge_model = Ridge(alpha=self.alpha)
        
        train_set = self.df_features.sample(frac=0.8, random_state=42)
    
        X_train = train_set[self.X_columns]
        y_train = train_set['point_diff']

        test_set = self.df_features.drop(train_set.index)

        # TODO: what can be toggled for a better outcome
        cv = KFold(n_splits=5, shuffle=True, random_state=42) 
        neg_mse_scores = cross_val_score(ridge_model, X_train, y_train, cv=cv, scoring="neg_mean_squared_error" )
        rmse_scores = np.sqrt(-neg_mse_scores) 
        r2_scores = cross_val_score(ridge_model, X_train, y_train, cv=cv, scoring="r2") 
        
        print("CV RMSE:", rmse_scores.mean(), "+/-", rmse_scores.std()) 
        print("CV R^2:", r2_scores.mean(), "+/-", r2_scores.std()) 

        #Fit the final model on all data:
        ridge_model.fit(X_train, y_train)

        y_test = ridge_model.predict(test_set[self.X_columns])

        plot_results(y_test, test_set[self.y_label].reset_index(drop=True))

    
    def train(self, train_set):
        X_train = train_set[self.X_columns]
        y_train = train_set[self.y_label]

        self.model.fit(X_train, y_train)

    def test(self, test_set):
        X_test = test_set[self.X_columns]
        y_test = test_set[self.y_label]

        y_pred = self.model.predict(X_test)

        plot_results(y_pred, y_test.reset_index(drop=True))

    #def predict(self):

