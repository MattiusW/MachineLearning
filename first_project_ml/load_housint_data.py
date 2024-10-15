import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tarfile
import urllib.request
from sklearn.preprocessing import OrdinalEncoder
from pathlib import Path
from zlib import crc32
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import rbf_kernel
from ClusterSimilarity import ClusterSimilarity
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector, make_column_transformer

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

def column_ratio(X):
    return X[:, [0]] / X[:, [1]]

def ratio_name(function_transformer, feature_names_in):
    return["ratio"] #features name

def ratio_pipeline():
    return make_pipeline(SimpleImputer(strategy="median"), FunctionTransformer(column_ratio, feature_names_out=ratio_name), StandardScaler())

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

    print("Check correlation:")
    # Make new attribiute data
    housing["pokoje_na_rodzine"] = housing["total_rooms"]/housing["households"]
    housing["wspolczynnik_sypialni"] = housing["total_bedrooms"]/housing["total_rooms"]
    housing["liczba_pokoi_na_dom"] = housing["population"]/housing["households"]

    # Select only numerical column
    housing_only_value_num = housing.select_dtypes(include=[np.number])

    # Check rank correlation every atribiute with mediana price house
    corr_matrix = housing_only_value_num.corr()
    print(corr_matrix["median_house_value"].sort_values(ascending=False))

    # Show correllation graphs
    attributes =["median_house_value", "median_income", "total_rooms", "housing_median_age"]

    scatter_matrix(housing[attributes], figsize=(12,8))

    # Primary data version
    housing = strat_train_set.drop("median_house_value", axis=1)
    housing_labels = strat_train_set["median_house_value"].copy()

    # Imputation data
    imputer = SimpleImputer(strategy="median")
    imputer.fit(housing_only_value_num)

    print("Imputer statistics: ", imputer.statistics_)
    print("Median values: ", housing_only_value_num.median().values)

    # Transform data to learning set
    X = imputer.transform(housing_only_value_num)
    # Put name column and index
    housing_tr = pd.DataFrame(X, columns=housing_only_value_num.columns,index=housing_only_value_num.index)

    # Transform string category to numeral
    housing_cat = housing[["ocean_proximity"]]
    ordinal_encoder = OrdinalEncoder()
    housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)

    # One-hot encoding
    cat_encoder = OneHotEncoder()
    housing_cat_1hot = cat_encoder.fit_transform(housing_cat)

    print("One-hot matrix: ", housing_cat_1hot.toarray())

    # Scaling
    min_max_scaler = MinMaxScaler(feature_range=(-1,1))
    housing_num_min_max_scaled = min_max_scaler.fit_transform(housing_only_value_num)

    # Standard mechanizm
    std_scaler = StandardScaler()
    housing_num_std_scaled = std_scaler.fit_transform(housing_only_value_num)

    target_scaler = StandardScaler()
    scaled_labels = target_scaler.fit_transform(housing_labels.to_frame())

    model = LinearRegression()
    model.fit(housing[["median_income"]], scaled_labels)
    some_new_data = housing[["median_income"]].iloc[:5]

    scaled_predictions = model.predict(some_new_data)
    predictions = target_scaler.inverse_transform(scaled_predictions)

    model = TransformedTargetRegressor(LinearRegression(), transformer=StandardScaler())
    model.fit(housing[["median_income"]], housing_labels)
    predictions = model.predict(some_new_data)

    log_transformer = FunctionTransformer(np.log, inverse_func=np.exp)
    log_pop = log_transformer.transform(housing[["population"]])

    rbf_transformer = FunctionTransformer(rbf_kernel, kw_args=dict(Y=[[35.]], gamma=0.1))
    age_simil_35 = rbf_transformer.transform(housing[["housing_median_age"]])
    sf_coords = 37.7749, -122.41
    sf_transformer = FunctionTransformer(rbf_kernel, kw_args=dict(Y=[sf_coords], gamma=0.1))
    sf_simil = sf_transformer.transform(housing[["latitude", "longitude"]])

    ratio_transformer = FunctionTransformer(lambda X: X[:, [0]] / X[:, [1]])
    print(ratio_transformer.transform(np.array([[1.,2.], [3., 4.]])))

    cluster_simil = ClusterSimilarity(n_cluster=10, gamma=1., random_state=42)
    similarities = cluster_simil.fit_transform(housing[["latitude", "longitude"]], sample_weight=housing_labels)

    print(f"Similarity: {similarities}")

    num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    housing_num_prepared = num_pipeline.fit_transform(housing_only_value_num)
    print(housing_num_prepared[:2].round(2))

    df_housing_num_prepared = pd.DataFrame(housing_num_prepared, columns=num_pipeline.get_feature_names_out(),index=housing_only_value_num.index)

    # Transform column
    num_attribs = ["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms", "population", "households", "median_income"]
    cat_attribs = ["ocean_proximity"]

    cat_pipeline = make_pipeline(SimpleImputer(strategy="most_frequent"),OneHotEncoder(handle_unknown="ignore"))

    #preprocessing = ColumnTransformer([("num", num_pipeline, num_attribs), ("cat", cat_pipeline, cat_attribs)])

    # Custom name column
    preprocessing = make_column_transformer((num_pipeline, make_column_selector(dtype_include=np.number)), (cat_pipeline, make_column_selector(dtype_include=object)),)

    housing_prepared = preprocessing.fit_transform(housing)

    log_pipeline = make_pipeline(SimpleImputer(strategy="median"), FunctionTransformer(np.log, feature_names_out="one-to-one"), StandardScaler())

    cluster_simil = ClusterSimilarity(n_cluster=10, gamma=1., random_state=42)
    default_num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())

    preprocessing = ColumnTransformer([("współczynnik_sypialni", ratio_pipeline(), ["total_bedrooms", "total_rooms"]),("pokoje_na_rodzine", ratio_pipeline(), ["total_rooms","households"]),("liczba_osób_na_dom", ratio_pipeline(), ["population", "households"]),("log", log_pipeline, ["total_bedrooms", "total_rooms", "population", "households", "median_income"]),("geo",cluster_simil, ["latitude", "longitude"]), ("cat", cat_pipeline,
                                                                                                                                                                                                                                                                                                                                                                                                                                make_column_selector(dtype_include=object)),],remainder=default_num_pipeline)

    housing_prepared = preprocessing.fit_transform(housing)
    print(housing_prepared.shape)
    print(preprocessing.get_feature_names_out())

    plt.show() # View graphs

if __name__ == "__main__":
    main()
