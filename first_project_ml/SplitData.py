import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class SplitData():
    def __init__(self):
        self.strat_train_set = []
        self.strat_test_set = []

    def split_data_frame_to_train_and_test_set(self, data_frame):
        data_frame["income_cat"] = pd.cut(data_frame["median_income"],bins=[0.,1.5,3.0,4.5,6.,np.inf], labels=[1,2,3,4,5])
        # Próbkowanie warstwowe za pomocą funkcji sklearn
        strat_train_set, strat_test_set = train_test_split(data_frame, test_size=0.2, stratify=data_frame["income_cat"],
                                                           random_state=42)
        self.strat_train_set = strat_train_set
        self.strat_test_set = strat_test_set

    def get_strat_train_set(self):
        return self.strat_train_set

    def get_strat_test_set(self):
        return self.strat_test_set