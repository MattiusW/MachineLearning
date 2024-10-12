from pathlib import Path
from zlib import crc32
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tarfile
import urllib.request

# Loading data for learn with github
def load_housing_data():
    tarball_path = Path("datasets/housing.tgz")
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path="datasets")
    return pd.read_csv(Path("datasets/housing/housing.csv"))

# Split to test data
def shuffle_and_split_data(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

# Split to unique data with hash id
def is_id_in_test_set(identifier, test_ratio):
    return crc32(np.int64(identifier)) < test_ratio * 2**32

def split_data_with_id_hash(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: is_id_in_test_set(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

def main():
    housing = load_housing_data()
    housing_with_id = housing.reset_index() # Add index column cuz data not have
    train_set, test_set = split_data_with_id_hash(housing_with_id, 0.2, "index")
    housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
    train_set, test_set = split_data_with_id_hash(housing_with_id, 0.2, "id")
    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
    # Set income mediana
    housing["income_cat"] = pd.cut(housing["median_income"],
            bins=[0.,1.5,3.0,4.5,6.,np.inf],
            labels=[1,2,3,4,5])
    housing["income_cat"].value_counts().sort_index().plot.bar(rot=0, grid=True)
    plt.xlabel("Kategoria dochodów")
    plt.ylabel("Liczba dystryktów")
    housing.hist(bins=50, figsize=(12,8)) # Set all data graphs 
    
    # Stratifield sampling
    splitter = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
    strat_splits = []
    for train_index, test_index in splitter.split(housing, housing["income_cat"]):
        strat_train_set_n = housing.iloc[train_index]
        strat_test_set_n = housing.iloc[test_index]
        strat_splits.append([strat_train_set_n, strat_test_set_n])
    
    # Check proportional data on example
    strat_train_set, strat_test_set = strat_splits[0]
    strat_train_set, strat_test_set = train_test_split(housing, test_size=0.2, stratify=housing["income_cat"], random_state=42)
    
    print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))

    # Delete columne income_cat
    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)
    
    # Copy data
    housing = strat_train_set.copy()

    # Visual data
    housing.plot(kind="scatter", x="longitude", y="latitude", grid=True, s=housing["population"]/ 100, label="population", c="median_house_value", cmap="jet", colorbar=True, legend=True, sharex=False, figsize=(10,7))

    plt.show() # View graphs

if __name__ == "__main__":
    main()
