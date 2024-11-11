import numpy as np
from zlib import crc32

class TestSet:

    def __init__(self, data, test_ratio, id_column=None):
        self.data = data
        self.test_ratio = test_ratio
        self.id_column = id_column
        self.save_data = None
        self.process_data()

    def process_data(self):
        train_set, test_set = self.split_data_with_id_hash(self.data, self.test_ratio, self.id_column)
        self.save_data = train_set, test_set        

    # Wybieranie losowo 20% części przykładów
    # UWAGA przy następnym uruchomieniu programu zostanie wygerenowany całkowicie inny zbiór testowy!
    @staticmethod
    def shuffle_and_split_data(data, test_ratio):
        shuffled_indices = np.random.permutation(len(data))
        test_set_size = int(len(data) * test_ratio)
        test_indices = shuffled_indices[:test_set_size]
        train_indices = shuffled_indices[test_set_size:]
        return data.iloc[train_indices], data.iloc[test_indices]

    # Obliczanie hasha każdego id aby zagwarantować stabilność zestawu testowego nawet po akutalizacji
    @staticmethod
    def is_id_in_test_set(indentifier, test_ratio):
        return crc32(np.int64(indentifier)) < test_ratio * 2**32

    def split_data_with_id_hash(self, data, test_ratio, id_column):
        ids = data[id_column]
        in_test_set = ids.apply(lambda id_: self.is_id_in_test_set(id_, test_ratio))
        return data.loc[~in_test_set], data.loc[in_test_set]


