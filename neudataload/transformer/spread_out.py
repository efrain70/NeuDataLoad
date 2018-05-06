"""Spread out matrix transformer class."""

from neudataload import spread_out_matrix
from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin


class SpreadOutMatrixTransformer(BaseEstimator, TransformerMixin):
    """Spread out a column with matrices in N*M features."""

    def __init__(self, columns=None, symmetric=True):
        """Instance a new SpreadOutMatrixTransformer.

        Args:
            columns: column names to be spread out (2D matrices)
            symmetric: if true remove elements in the upper diagonal.

        """
        self.columns = columns
        self.symmetric = symmetric

    def fit(self, *args, **kwargs):
        """Fit the SpreadOutMatrixTransformer transformer.

        Args:
            *args: not used
            **kwargs: not used

        Returns:
            self

        """
        return self

    def transform(self, x):
        """Spread out the matrices in N*M new features.

        Args:
            x: array of shape [n_samples, n_features]

        Returns
            array of shape n_features + N*M*n_columns
            with the content of the matrices spread out in
            new features

        """
        df = x
        is_df = False

        if self.columns:
            if not isinstance(x, DataFrame):
                df = DataFrame(data=x)
                is_df = True
            df = spread_out_matrix(df, self.columns, keep_matrix=False,
                                   symmetric=self.symmetric)

        return df if not is_df else df.values
