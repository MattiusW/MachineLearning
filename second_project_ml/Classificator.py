from sklearn.linear_model import SGDClassifier

class Classificator():
    def __init__(self):
        self.sgd_predict = None

    def sgd_model(self, data1, data2, X):
        sgd_clf = SGDClassifier(random_state=42)
        sgd_clf.fit(data1,data2)
        self.sgd_predict = sgd_clf.predict([X])
