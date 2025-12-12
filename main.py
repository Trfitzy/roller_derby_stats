import logging
from argparse import ArgumentParser
import import_ipynb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, KFold

import warnings
from model import Model

warnings.filterwarnings('ignore')

model_help = """Derby Prediction Model: Used to train, test, and predict skater combinations"""

def parse_args():
    arg_parser = ArgumentParser(
        prog="Derby Prediction Model", usage="Used to train, test, and predict skater combinations"
    )
    arg_parser.add_argument(
        "action", type=str, choices=["train", "test", 'predict'], help=model_help)
    arg_parser.add_argument(
        "data_path", type=str)
    
    return arg_parser.parse_args()


if __name__ == "__main__":
    input_args = parse_args()

    model = Model('VL')
    model.preprocess_data()
    model.train_test_known_data()

    
    #except FileNotFoundError:
    #    print("The file path " + input_args.data_path + " does not contain train & test folders or those folders are missing files.")

  
