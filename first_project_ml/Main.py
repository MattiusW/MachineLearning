import numpy as np
import pandas as pd
from reportlab.graphics.transform import inverse
from tensorflow.python.autograph.converters.functions import FunctionTransformer

import load_housint_data
from MLmodels import MLmodels
from Prediction import Prediction
from PrepareData import PrepareDate
from SplitData import SplitData
from TransformPipeline import TransformPipeline

def main():
    housing = load_housint_data.load_housing_data()
    housing_with_sklearn = SplitData()
    housing_with_sklearn.split_data_frame_to_train_and_test_set(housing)
    strat_test_set = housing_with_sklearn.get_strat_test_set()
    strat_train_set = housing_with_sklearn.get_strat_train_set()
    # Kopia danych
    # housing_strat_train_copy = strat_train_set.copy()
    housing_strat_train_labales_copy = strat_train_set["median_house_value"].copy()

    housing_prepare_data = PrepareDate()
    housing_prepare_data.imputation_data(strat_train_set)
    get_X = housing_prepare_data.get_X()
    housing = strat_train_set
    housing_cat_encoded = housing_prepare_data.category_to_num(housing[["ocean_proximity"]])
    housing[["ocean_proximity"]] = housing_cat_encoded
    one_hot_encoded = housing_prepare_data.hot_encode(housing_cat_encoded)

    # Skalowanie kodowanych danych
    housing_min_max_scaled = housing_prepare_data.scaling_data(housing_cat_encoded)


    # Predykcja modelu LinearRegression
    model = Prediction()
    model.train_linear_model(housing[["median_income"]], housing_strat_train_labales_copy.to_frame())


    # # Pipeline
    transform_pipeline = TransformPipeline(housing_cat_encoded)
    transform_pipeline.call_pipeline()
    transform_pipeline.transform_column_data(housing)
    preprocessing = transform_pipeline.get_prepared_data()
    pre_data = TransformPipeline(housing)
    # processing = pre_data.log_pipeline_prepared_data()
    # processing.fit_transform(housing)


    #
    # # ML Models
    ml_lin_reg = MLmodels()
    # lin_reg = ml_lin_reg.line_reg_model(prepare_data, housing, housing_strat_train_labales_copy)
    #
    # lin_reg_predictions = lin_reg.predict(housing)
    # print("Linear predicts: ", lin_reg_predictions[:5].round(-2))

if __name__ == "__main__":
    main()