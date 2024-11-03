from pandas.core.common import random_state
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.svm import SVC


class Classificator():
    def __init__(self):
        self.sgd_clf = SGDClassifier(random_state=42)
        self.dummy_clf = DummyClassifier()
        self.forest_clf = RandomForestClassifier(random_state=42)
        self.svm_clf = SVC(random_state=42)

    def sgd_clf_model(self, data_x, data_y, X):
        self.sgd_clf.fit(data_x, data_y)
        sgd_predict = self.sgd_clf.predict([X])
        return sgd_predict

    def dummy_clf_model(self, data_x, data_y, X):
        self.dummy_clf.fit(data_x, data_y)
        dummy_predict = self.dummy_clf.predict(X)
        return dummy_predict

    def forest_clf_model(self, data_x, data_y):
        predict_proba = cross_val_predict(self.forest_clf, data_x, data_y, cv=3, method="predict_proba")
        return predict_proba

    def get_sgd_clf_model(self):
        return self.sgd_clf

    def get_dummy_clf_model(self):
        return self.dummy_clf


