from binascii import crc32
import numpy as np

class PrepareDate():
    def __init__(self, loadData):
        self.loadData = loadData

    # Przetasuj i podziel dane na zestawy testowe i treningowe
    def shuffle_and_split_data(loadData, test_ratio):
        shuffled_indices = np.random.permutation(len(loadData))
        test_set_size = int(len(loadData) * test_ratio)
        test_indices = shuffled_indices[:test_set_size]
        train_indices = shuffled_indices[test_set_size:]
        return loadData.iloc[train_indices], loadData.iloc[test_indices]

    # Sprawdź czy rekord powinien trafić do zestawu testowego
    def is_id_in_test_set(identifier, test_ratio):
        return crc32(np.int64(identifier)) < test_ratio * 2 ** 32

    def split_data_with_id_hash(self, data, test_ratio, id_column):
        ids = data[id_column]
        in_test_set = ids.apply(lambda id_: self.is_id_in_test_set(id_, test_ratio))
        return data.loc[~in_test_set], data.loc[in_test_set]

    # String override
    def __str__(self):
        return f"Prepared data: {self.loadData}"
