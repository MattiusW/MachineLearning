from pathlib import Path
import numpy as np
import pandas as pd
import tarfile
import urllib.request
from sklearn.model_selection import train_test_split
from TestSet import TestSet

# Wczytanie danych
@staticmethod
def load_housing_data():
    tarball_path = Path("datasets/housing.tgz")
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path="datasets")
    return pd.read_csv(Path("datasets/housing/housing.csv"))

# Rozdzelenie danych na zbior testowy
@staticmethod
def load_test_set():
    housing = load_housing_data()
    housing_with_id = housing.reset_index()
    # Wygenerowanie id ze współrzędnych geograficznych
    housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
    test_set_instance = TestSet(housing_with_id, test_ratio=0.2, id_column="id")
    train_set, test_set = test_set_instance.save_data
    return train_set, test_set

def load_spliter_test_set():
    housing = load_housing_data()
    housing["income_cat"] = pd.cut(housing["median_income"], bins=[0., 1.5, 3.0, 4.5, 6., np.inf], labels=[1,2,3,4,5])
    strat_train_set, strat_test_set = train_test_split(housing, test_size=0.2, stratify=housing["income_cat"], random_state=42)
    return strat_train_set, strat_test_set

def delete_inf_column(strat_train_set, strat_test_set):
    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)