from pandas.core.common import random_state
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
import load_housint_data
from PrepareData import PrepareDate
from SplitData import SplitData


def main():
    housing = load_housint_data.load_housing_data()
    housing_with_sklearn = SplitData()
    housing_with_sklearn.split_data_frame_to_train_and_test_set(housing)
    strat_test_set = housing_with_sklearn.get_strat_test_set()
    strat_train_set = housing_with_sklearn.get_strat_train_set()
    # Kopia danych
    # housing_strat_train_copy = strat_train_set.copy()
    # housing_strat_train_labales_copy = strat_train_set["median_house_value"].copy()
    housing_prepare_data = PrepareDate()
    housing_imputation_data = housing_prepare_data.imputation_data(strat_train_set)
    get_X = housing_prepare_data.get_X()
    housing = strat_train_set
    housing_cat_encoded = housing_prepare_data.category_to_num(housing[["ocean_proximity"]])
    one_hot_encoded = housing_prepare_data.hot_encode(housing_cat_encoded)


    print("Próbkowanie za pomocą funkcji wbudowanej w bibliotekę: ")
    print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))

    print("Test transformatorów: ")
    print(get_X)
    print("Num data: ")
    print(housing_prepare_data)
    print(housing[["ocean_proximity"]].head(8))
    print(housing_cat_encoded[:8])
    print("One hot: ", one_hot_encoded.toarray())

if __name__ == "__main__":
    main()