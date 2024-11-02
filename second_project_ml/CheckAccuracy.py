from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.model_selection import cross_val_score, cross_val_predict


class CheckAccuracy():
    def __init__(self):
        self.predict = None
        self.decision_score = None

    def model_cross_value_precision(self, model_to_check, data_x, data_y):
        result = cross_val_score(model_to_check, data_x, data_y, cv=3, scoring="accuracy")
        return result

    def model_confusion_matrix(self, model_to_check, data_x, data_y):
        self.predict = cross_val_predict(model_to_check, data_x, data_y, cv=3)
        cm = confusion_matrix(data_y, self.predict)
        return cm

    def model_decision_score(self, model_to_score, data_x, data_y):
        self.decision_score = cross_val_predict(model_to_score, data_x, data_y, cv=3, method="decision_function")
        precision_rec = precision_recall_curve(data_y, self.decision_score)
        return precision_rec
