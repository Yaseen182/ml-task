from sklearn.datasets import load_iris
import pandas as pd

def load_data():
    data = load_iris()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")
    return X, y
