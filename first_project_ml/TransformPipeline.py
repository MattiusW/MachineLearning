import numpy as np
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


class TransformPipeline():
    def __init__(self, data):
        self.num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
        self.cat_pipeline = make_pipeline(SimpleImputer(strategy="median"), OneHotEncoder(handle_unknown="ignore"))
        self.data = data
        self.num_prepared = []
        self.preprocessing = None
        self.data_prepared = None

    def call_pipeline(self):
        self.num_prepared = self.num_pipeline.fit_transform(self.data)
        return self.num_prepared

    def transform_column_data(self, data):
        self.preprocessing = make_column_transformer(
            (self.num_pipeline, make_column_selector(dtype_include=np.number)),
            (self.cat_pipeline, make_column_selector(dtype_include=object))
        )
        self.data_prepared = self.preprocessing.fit_transform(data)