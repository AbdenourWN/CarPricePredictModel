import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class TargetEncoder(BaseEstimator, TransformerMixin):
    """
    Encodes categorical features by replacing them with the mean of the target variable.
    This is a powerful technique for high-cardinality features.
    """
    def __init__(self, handle_unknown='mean'):
        self.handle_unknown = handle_unknown

    def fit(self, X, y):
        """
        Learns the mapping from each category to the mean of the target.
        Args:
            X (pd.DataFrame): The feature data.
            y (pd.Series): The target data.
        """
        data = pd.concat([X, y], axis=1)
        self.mappers_ = {}
        self.target_mean_ = y.mean()
        for col in X.columns:
            self.mappers_[col] = data.groupby(col)[y.name].mean()
        return self

    def transform(self, X, y=None):
        """
        Applies the learned mapping to the feature data.
        """
        X_copy = X.copy()
        for col in X_copy.columns:
            X_copy[col] = X_copy[col].map(self.mappers_[col])
            if self.handle_unknown == 'mean':
                X_copy[col] = X_copy[col].fillna(self.target_mean_)
        return X_copy