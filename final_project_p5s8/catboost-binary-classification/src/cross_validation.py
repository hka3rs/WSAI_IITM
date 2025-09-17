from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import KFold
from sklearn.metrics import hinge_loss
import numpy as np

def cross_validate_model(X, y, n_splits=10):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    losses = []

    for train_index, val_index in kf.split(X):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        model = CatBoostClassifier(loss_function='Hinge', iterations=1000, learning_rate=0.1, depth=6, verbose=0)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        loss = hinge_loss(y_val, y_pred)
        losses.append(loss)

    return np.mean(losses), np.std(losses)