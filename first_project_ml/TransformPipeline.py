from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler


class TransformPipeline():
    def __init__(self, data):
        self.num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
        self.data = data
        self.num_prepered = []

    def call_pipeline(self):
        self.num_prepered = self.num_pipeline.fit_transform(self.data)
        return self.num_prepered

