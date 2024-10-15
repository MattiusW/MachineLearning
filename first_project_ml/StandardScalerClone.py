from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

class StandardScalerClone(BaseEstimator, TransformerMixin):
    def __init__(sefl, with_mean=True):
        self.with_mean = with_mean

    def fit(self, X, y=None):
        X = check_array(X) # Check if table end with float numbers
        self.mean = X.mean(axis=0)
        self.scale = X.std(axis=0)
        self.n_features_in_ = X.shape[1]
        return self # ALWAYS RETURN SELF!

    def transform(self, X):
        check_is_fitted(self)
        X = check_array(X)
        assert self.n_features_in_ == X.shape[1]
        if self.with_mean:
            X = X - self.mean_
            return X / self.scale_
