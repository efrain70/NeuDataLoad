"""Feature selection matrix transformer class."""

from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureMatrixTransformer(BaseEstimator, TransformerMixin):
    """Select the features in data with matrix."""

    def __init__(self, columns=None, matrix_columns=None):
        """Instance a new FeatureMatrixTransformer.

        Args:
            columns: column names to be selected
            matrix_columns: features created with spread out from 2D items.

        """
        self.columns = columns
        self.matrix_columns = matrix_columns
        self.regex = None

    def fit(self, *args, **kwargs):
        """Fit the FeatureMatrixTransformer transformer.

        Args:
            *args: not used
            **kwargs: not used

        Returns:
            self

        """
        self.columns = [str(x) for x in self.columns] \
            if self.columns is not None else []
        self.matrix_columns = [str(x) for x in self.matrix_columns] \
            if self.matrix_columns is not None else []

        self.regex = '|'.join(list(self.matrix_columns) + list(self.columns))

        return self

    def transform(self, x):
        """Reduce x with only the selected features.

        Args:
            x: array of shape [n_samples, n_features]


        Returns:
            array of shape [n_samples, n_selected_features]
            with only the selected features.

        """
        df = x
        is_df = False

        if not isinstance(x, DataFrame):
            df = DataFrame(data=x)
            is_df = True

        if self.regex:
            df = df.filter(regex=self.regex, axis=1)

        return df if not is_df else df.values
