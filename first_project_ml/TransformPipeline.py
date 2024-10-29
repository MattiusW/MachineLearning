import numpy as np
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer

from ClusterSimilarity import ClusterSimilarity


class TransformPipeline():
    def __init__(self, data):
        self.num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
        self.cat_pipeline = make_pipeline(SimpleImputer(strategy="median"), OneHotEncoder(handle_unknown="ignore"))
        self.data = data
        self.num_prepared = []
        self.preprocessing = None
        self.data_prepared = None
        self.log_pipeline_preprocessing = None

    def call_pipeline(self):
        self.num_prepared = self.num_pipeline.fit_transform(self.data)
        return self.num_prepared

    def transform_column_data(self, data):
        self.preprocessing = make_column_transformer(
            (self.num_pipeline, make_column_selector(dtype_include=np.number)),
            (self.cat_pipeline, make_column_selector(dtype_include=object))
        )
        self.data_prepared = self.preprocessing.fit_transform(data)

    def get_prepared_data(self):
        return self.data_prepared

    def column_ratio(self, X):
        return X[:, [0]] / X[:, [1]]

    def ratio_name(self, function_transformer, feature_names_in):
        return ["ratio"]  # features name

    def ratio_pipeline(self):
        return make_pipeline(SimpleImputer(strategy="median"),
                             FunctionTransformer(TransformPipeline.column_ratio, feature_names_out=TransformPipeline.ratio_name),
                             StandardScaler())

    def log_pipeline_prepared_data(self):
        return make_pipeline(SimpleImputer(strategy="median"),
                             FunctionTransformer(np.log, feature_names_out="one-to-one"), StandardScaler())
        self.cluster_simil = ClusterSimilarity(n_cluster=10, gamma=1., random_state=42)
        self.default_num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
        self.log_pipeline_preprocessing = ColumnTransformer([
            ("współczynnik_sypialni", self.ratio_pipeline(), ["total_bedrooms", "total_rooms"]),
            ("pokoje_na_rodzine", self.ratio_pipeline(), ["total_rooms","households"]),
            ("liczba_osób_na_dom", self.ratio_pipeline(), ["population", "households"]),
            ("log", self.log_pipeline, ["total_bedrooms", "total_rooms", "population", "households", "median_income"]),
            ("geo",self.cluster_simil, ["latitude", "longitude"]),
            ("cat", self.cat_pipeline, make_column_selector(dtype_include=object)),],remainder=default_num_pipeline)

