from sklearn.dummy import DummyClassifier
from sklearn.linear_model import SGDClassifier
from softwareproperties.AptAuth import dummy


class Classificator():
    def __init__(self):
        self.sgd_clf = SGDClassifier(random_state=42)
        self.dummy_clf = DummyClassifier()

    def sgd_clf_model(self, data1, data2, X):
        self.sgd_clf.fit(data1,data2)
        sgd_predict = self.sgd_clf.predict([X])
        return sgd_predict

    def dummy_clf_model(self, data1, data2, X):
        self.dummy_clf.fit(data1, data2)
        dummy_predict = self.dummy_clf.predict(X)
        return dummy_predict

    def get_sgd_clf_model(self):
        return self.sgd_clf

    def get_dummy_clf_model(self):
        return self.dummy_clf