import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, MinMaxScaler


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

    def category_to_num(self, cat_to_num):
        ordinal_encoder = OrdinalEncoder()
        cat_encoded = ordinal_encoder.fit_transform(cat_to_num)
        return cat_encoded

    def hot_encode(self, cat_to_hot_encode):
        return OneHotEncoder().fit_transform(cat_to_hot_encode)

    def scaling_data(self, data_to_scaling):
        min_max_scaler = MinMaxScaler(feature_range=(-1,1))
        num_min_max_scaled = min_max_scaler.fit_transform(data_to_scaling)
        return num_min_max_scaled

    def get_X(self):
        return self.X

    def __str__(self):
        return f"Data Frame: {self.data_num}"