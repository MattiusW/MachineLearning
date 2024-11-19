import numpy as np
from sklearn.impute import SimpleImputer

class PrepareDate:
    def __init__(self):
        pass

    def column_string_to_int(self, data):
        housing_num = data.select_dtypes(include=[np.number])
        return housing_num
    
    def impute_data(self, data):
        imputer = SimpleImputer(strategy="median")
        imputer.fit(data)
        X = imputer.transform(data)
        return X