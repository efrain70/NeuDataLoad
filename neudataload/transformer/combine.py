"""Combine matrix transformer class."""


from neudataload import combine_matrix
import numpy as np
from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin


class CombineMatrixTransformer(BaseEstimator, TransformerMixin):
    """Combine the values of matrix in a column using a function."""

    def __init__(self, column_name=None, columns=None, op=np.mean):
        """Instance a new CombineMatrixTransformer.

        Args:
            column_name: column name
            columns: columns to be combined
            op: function to apply

        """
        self.columns = columns
        self.column_name = column_name
        self.op = op

    def fit(self, *args, **kwargs):
        """Fit the CombineMatrixTransformer transformer.

        Args:
            *args: not used
            **kwargs: not used

        Returns:
            self

        """
        return self

    def transform(self, x):
        """Combine the matrix in X with the selected function.

        Args:
            x: array of shape [n_samples, n_features]

        Returns:
            array of shape [n_samples, n_features + 1]
            with a extra featured with the combined matrix.

        """
        df = x
        is_df = False

        if self.columns and self.column_name:
            if not isinstance(x, DataFrame):
                columns = [str(c_i) for c_i in range(x.shape[1])]
                self.column_name = str(x.shape[1] + 1)

                df = DataFrame(data=x, columns=columns)

                is_df = True

            df = combine_matrix(
                df, columns=self.columns,
                column_result=self.column_name, func=self.op)

        import ipdb; ipdb.set_trace()

        return df if not is_df else df.as_matrix(
            columns=df.columns.sort_values())
