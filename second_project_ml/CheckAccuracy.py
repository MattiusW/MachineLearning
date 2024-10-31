from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, cross_val_predict


class CheckAccuracy():
    def __init__(self):
        self.predict = None

    def model_cross_value_precision(self, model_to_check, data1, data2):
        result = cross_val_score(model_to_check, data1, data2, cv=3, scoring="accuracy")
        return result

    def model_confusion_matrix(self, model_to_check, data1, data2):
        self.predict = cross_val_predict(model_to_check, data1, data2, cv=3)
        cm = confusion_matrix(data2, self.predict)
        return cm
