import pandas as pd
from sklearn import datasets

def import_data():
    iris = datasets.load_iris()
    iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    return iris_df

print(import_data())