from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


class Prediction():
    def __init__(self):
        self.predictions = None
    #
    # def transform_target_regressor(self, data, data_labels, some_new_data):
    #     model = TransformedTargetRegressor(LinearRegression(), transformer=StandardScaler())
    #     model.fit(data, data_labels)
    #     self.predictions = model.predict(some_new_data)

    def train_linear_model(self, labels, data):
        target_scaler = StandardScaler()
        scaled_labels = target_scaler.fit_transform(labels)

        model = LinearRegression()
        model.fit(data, scaled_labels)
        some_new_data = data

        scaled_predictions = model.predict(some_new_data)
        self.predictions = target_scaler.inverse_transform(scaled_predictions)