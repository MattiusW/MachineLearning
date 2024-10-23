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
    housing_strat_train_copy = strat_train_set.copy()
    housing_strat_train_labales_copy = strat_train_set["median_house_value"].copy()

    print("Próbkowanie za pomocą funkcji wbudowanej w bibliotekę: ")
    print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))

if __name__ == "__main__":
    main()