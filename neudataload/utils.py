import numpy as np

def binarize_matrix(data_frame, columns, threshold=0):

    for column in columns:
        data_frame[column] = data_frame[
            data_frame[column].notnull()][column].apply(
            lambda x: 1 * (x > threshold))

    return data_frame


def combine_matrix(data_frame, columns, column_result, func=np.mean):
    compressed = [data_frame[c].values for c in columns]

    values_result = np.asarray(func(compressed, axis=0))
    df = data_frame.assign(**{column_result: values_result})

    return df
