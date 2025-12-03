import pandas as pd
from sklearn.preprocessing import StandardScaler

def handle_missing_values(X):
    return X.fillna(X.mean())

def remove_outliers(X, y):
    Q1 = X.quantile(0.25)
    Q3 = X.quantile(0.75)
    IQR = Q3 - Q1
    mask = ~((X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR))).any(axis=1)
    return X[mask], y[mask]

def normalize_data(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled
