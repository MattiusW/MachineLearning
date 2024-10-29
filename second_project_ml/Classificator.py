from sklearn.linear_model import SGDClassifier

class Classificator():
    def sgd_model(self, data1, data2, X):
        sgd_clf = SGDClassifier(random_state=42)
        sgd_clf.fit(data1,data2)
        sgd_predict = sgd_clf.predict([X])
        return sgd_predict
