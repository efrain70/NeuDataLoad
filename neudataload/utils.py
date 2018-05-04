def binarize_matrix(data_frame, columns, threshold=0):

    for column in columns:
        data_frame[column] = data_frame[
            data_frame[column].notnull()][column].apply(
            lambda x: 1 * (x > threshold))

    return data_frame
