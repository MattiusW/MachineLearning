import load_housint_data
from PrepareData import PrepareDate


def main():
    housing = load_housint_data.load_housing_data()
    housing_with_id = PrepareDate(housing)
    print(housing_with_id)

if __name__ == "__main__":
    main()