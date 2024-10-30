from sklearn.model_selection import cross_val_score


class CheckAccuracy():
    def model_cross_value_precision(self, model_to_check, data1, data2):
        result = cross_val_score(model_to_check, data1, data2, cv=3, scoring="accuracy")
        return result
