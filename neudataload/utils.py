"""neudataload.utils.

Utils for managing matrices in data frames.
"""

import itertools

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer


def all_combinations(iterable):
    """Create a list of interation with all the combinations.

    Args:
        iterable: iterable with the elements

    Returns:
        iterable of tuples with the all combinations of 0-N sizes.

    """
    return itertools.chain.from_iterable(
        itertools.combinations(iterable, i + 1)
        for i in range(len(iterable)))


def binarize_matrix(data_frame, columns, threshold=0):
    """Binarize with 1/0 the content of the matrices located in columns.

    The matrices are located in the columns of a data frame, if their values
    is bigger than threshold, 1 if not 0.

    Args:
        data_frame: dataframe with matrices in columns
        columns: name of the columns
        threshold: value of threshold

    Returns:
        Data frame with its matrices binarized with 1 and 0

    """
    for column in columns:
        data_frame[column] = data_frame[
            data_frame[column].notnull()][column].apply(
            lambda x: 1 * (x > threshold))

    return data_frame


def combine_matrix(data_frame, columns, column_result, func=np.mean):
    """Mix matrices located in colunms of a data frame using a function.

    Args:
        data_frame: data frame with matrices.
        columns: columns of data frame where matrices ar located
        column_result: column of data frame for allocation the result.
        func: function for merging the values of the matrices.

    Returns:
        Data frame with extra new column that contains a combination of
        matrices.

    """
    compressed = [data_frame[c].values for c in columns]

    values_result = np.asarray(func(compressed, axis=0))
    df = data_frame.assign(**{column_result: values_result})

    return df


def spread_out_matrix(data_frame, columns, symmetric=True, keep_matrix=False):
    """Spread out a list of matrix located in columns of a data frame.

    Args:
        data_frame: data frame with matrices in columns
        columns: column names
        symmetric: if True, remove the repeated half of the matrix.
        keep_matrix: if False, keep the original column with the matrices.

    Returns:
        A data frame with a new column for each coordinate of the matrix and
        for each matrix.

    """
    df = data_frame

    new_dfs = list()

    for column in columns:
        reshaped = df[df[column].notnull()][column].apply(
            lambda x: x.reshape(-1))

        max_dim = int(np.sqrt(reshaped.apply(len).max()))

        values = reshaped.values.tolist()
        new_columns = ['{}_{}_{}'.format(column, str(x), str(y))
                       for x in list(range(0, max_dim))
                       for y in list(range(0, max_dim))]

        df_reshaped = pd.DataFrame(
            values, columns=new_columns, index=reshaped.index)

        if symmetric:
            no_duplicated = ['{}_{}_{}'.format(column, str(x), str(y))
                             for x in list(range(0, max_dim))
                             for y in list(range(0, x))]

            df_reshaped = df_reshaped.filter(items=no_duplicated)
        new_dfs.append(df_reshaped)

    if not keep_matrix:
        df = df.drop(columns=list(columns))

    return pd.concat([df, ] + new_dfs, axis=1)


def get_multilabel(data_frame, column, groups):
    """Convert a class attribute to a binary matrix.

    By indicating the presence of a class label in the instance with 0,1.

    Args:
        data_frame: data frame with the column
        column: column to extract the feature values
        groups: dictionary as key the class, value list of groups

    Returns:
       Binary data frame with the presence of the label. Class names as
       column names.

    """
    binarizer = MultiLabelBinarizer()

    y_multi = data_frame[column].apply(lambda x: groups[x])
    values = binarizer.fit_transform(y_multi)

    return pd.DataFrame(data=values, columns=binarizer.classes_,
                        index=data_frame.index)
