from LoadData import *


def main():
    train_set, test_set = load_test_set()
    strat_train_set, strat_test_set = load_spliter_test_set()
    test = strat_test_set["income_cat"].value_counts() / len(strat_test_set)
    print(test)

    # Usunac kolumne inf
    delete_inf_column(strat_train_set, strat_test_set)

if __name__ == "__main__":
    main()