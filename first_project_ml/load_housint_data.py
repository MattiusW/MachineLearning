from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tarfile
import urllib.request

def load_housing_data():
    tarball_path = Path("datasets/housing.tgz")
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path="datasets")
    return pd.read_csv(Path("datasets/housing/housing.csv"))

def main():
    housing = load_housing_data()
    housing.hist(bins=50, figsize=(12,8))
    plt.show()

if __name__ == "__main__":
    main()
