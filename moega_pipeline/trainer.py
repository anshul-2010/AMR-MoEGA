import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_features_and_labels(features_csv: str, labels_csv: str):
    X = pd.read_csv(features_csv, index_col=0)
    y = pd.read_csv(labels_csv, index_col=0)
    # Ensure order: X rows align to y rows by index if possible
    if X.shape[0] == y.shape[0]:
        return X.values, y.values.ravel()
    else:
        # If mismatch, assume samples correspond by order
        return X.values, y.values.ravel()


def train_val_split(X, y, test_size=0.2, random_state=0, stratify=True):
    if stratify:
        return train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )
    else:
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
