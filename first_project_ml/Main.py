from pandas.core.common import random_state
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
import load_housint_data
from PrepareData import PrepareDate
from PrepareDataV2 import PrepareDataV2


def main():
    housing = load_housint_data.load_housing_data()
    housing_with_sklearn = PrepareDataV2()
    housing_with_sklearn.preparing_data(housing)
    strat_test_set = housing_with_sklearn.get_strat_test_set()

    print("Próbkowanie za pomocą funkcji wbudowanej w bibliotekę: ")
    print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))

if __name__ == "__main__":
    main()