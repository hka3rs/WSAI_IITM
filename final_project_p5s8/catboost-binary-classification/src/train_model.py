# File: /catboost-binary-classification/catboost-binary-classification/src/train_model.py

import pandas as pd
from catboost import CatBoostClassifier, Pool
import joblib

def train_catboost_model(X, y, params=None):
    if params is None:
        params = {
            'iterations': 1000,
            'learning_rate': 0.1,
            'depth': 6,
            'loss_function': 'Logloss',
            'eval_metric': 'AUC',
            'random_seed': 42,
            'verbose': 100
        }
    
    model = CatBoostClassifier(**params)
    model.fit(X, y)
    return model

def save_model(model, filename):
    joblib.dump(model, filename)