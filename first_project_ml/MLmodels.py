from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline


class MLmodels():
    def __init__(self):
        self.lin_reg = None


    def line_reg_model(self, preprocessing_data, data, data_labels):
        self.lin_reg = make_pipeline(preprocessing_data, LinearRegression())
        self.lin_reg.fit_transform(data, data_labels)