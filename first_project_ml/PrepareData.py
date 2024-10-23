import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

class PrepareDate():
    def __init__(self):
        self.X = None
        self.data_num = None

    def imputation_data(self, data_frame):
        imputer = SimpleImputer(strategy="median")
        data_num = data_frame.select_dtypes(include=[np.number])
        imputer.fit(data_num)
        imputer_X = imputer.transform(data_num)
        pd.DataFrame(imputer_X, columns=data_num.columns, index=data_num.index)
        self.data_num = data_num
        self.X = imputer_X

    def get_X(self):
        return self.X

    def __str__(self):
        return f"Data Frame: {self.data_num}"