"""neudataload.profiles.

Profiles for loading files with the data into a dataframe.
"""

import logging
import os

import numpy as np
import pandas

from .utils import binarize_matrix, combine_matrix, spread_out_matrix

DIR_CONFIGURATION = {
    # "CONTROLS",
    # "CONTROLS/ADJACENCY_MATRICES_GRAPH",
    # "CONTROLS/ADJACENCY_MATRICES_GRAPH/DTI_indices",

    "CONTROLS/ADJACENCY_MATRICES_GRAPH/DTI_indices/FA":
        ('_FA_factor.csv', 'DTI_FA'),
    "CONTROLS/ADJACENCY_MATRICES_GRAPH/DTI_indices/L1":
        ('_L1_factor.csv', 'DTI_L1'),
    "CONTROLS/ADJACENCY_MATRICES_GRAPH/DTI_indices/MD":
        ('_MD_factor.csv', 'DTI_MD'),
    "CONTROLS/ADJACENCY_MATRICES_GRAPH/DTI_indices/RX":
        ('_RX_factor.csv', 'DTI_RX'),
    "CONTROLS/ADJACENCY_MATRICES_GRAPH/RAW":
        ('_weight_factor.csv', 'RAW'),

    # "FIS",
    # "FIS/ADJACENCY_MATRICES_GRAPH",
    # "FIS/ADJACENCY_MATRICES_GRAPH/DTI_indices",

    "FIS/ADJACENCY_MATRICES_GRAPH/DTI_indices/FA":
        ('_FA_factor.csv', 'DTI_FA'),
    "FIS/ADJACENCY_MATRICES_GRAPH/DTI_indices/L1":
        ('_L1_factor.csv', 'DTI_L1'),
    "FIS/ADJACENCY_MATRICES_GRAPH/DTI_indices/MD":
        ('_MD_factor.csv', 'DTI_MD'),
    "FIS/ADJACENCY_MATRICES_GRAPH/DTI_indices/RX":
        ('_RX_factor.csv', 'DTI_RX'),

    "FIS/ADJACENCY_MATRICES_GRAPH/FUNC":
        ('_r_matrix.csv', 'FUNC'),
    "FIS/ADJACENCY_MATRICES_GRAPH/LS":
        ('_matrix_LS.csv', 'LS'),
    "FIS/ADJACENCY_MATRICES_GRAPH/RAW":
        ('_weight_factor.csv', 'RAW'),

    # "PREDICTORS",
    # "PREDICTORS/ADJACENCY_MATRICES_GRAPH",
    # "PREDICTORS/ADJACENCY_MATRICES_GRAPH/DTI_indices",

    "PREDICTORS/ADJACENCY_MATRICES_GRAPH/DTI_indices/FA.csv":
        ('_FA_factor', 'DTI_FA'),
    "PREDICTORS/ADJACENCY_MATRICES_GRAPH/DTI_indices/L1.csv":
        ('_L1_factor', 'DTI_L1'),
    "PREDICTORS/ADJACENCY_MATRICES_GRAPH/DTI_indices/MD.csv":
        ('_MD_factor', 'DTI_MD'),
    "PREDICTORS/ADJACENCY_MATRICES_GRAPH/DTI_indices/RX.csv":
        ('_RX_factor', 'DTI_RX'),

    "PREDICTORS/ADJACENCY_MATRICES_GRAPH/LS":
        ('_matrix_LS.csv', 'LS'),
    "PREDICTORS/ADJACENCY_MATRICES_GRAPH/RAW":
        ('_weight_factor.csv', 'RAW'),

    # "REESCAN",
    # "REESCAN/ADJACENCY_MATRICES_GRAPH",
    # "REESCAN/ADJACENCY_MATRICES_GRAPH/DTI_indices",

    "REESCAN/ADJACENCY_MATRICES_GRAPH/DTI_indices/FA":
        ('_FA_factor.csv', 'DTI_FA'),
    "REESCAN/ADJACENCY_MATRICES_GRAPH/DTI_indices/L1":
        ('_L1_factor.csv', 'DTI_L1'),
    "REESCAN/ADJACENCY_MATRICES_GRAPH/DTI_indices/MD":
        ('_MD_factor.csv', 'DTI_MD'),
    "REESCAN/ADJACENCY_MATRICES_GRAPH/DTI_indices/RX":
        ('_RX_factor.csv', 'DTI_RX'),

    "REESCAN/ADJACENCY_MATRICES_GRAPH/LS":
        ('_matrix_LS.csv', 'LS'),
    "REESCAN/ADJACENCY_MATRICES_GRAPH/RAW":
        ('_weight_factor.csv', 'RAW'),
}


class NeuProfiles(object):
    """Main class for loading profiles and matrix in sub directories."""

    def __init__(self, profiles_path, profiles_filename, format_file='xls'):
        """Initialize a new profile configuration.

        Args:
            profiles_path: path with the data for the profiles in csv format.
            profiles_filename: name of the main file with the list of profiles.
            format_file: format of the main file, xls for excel files.

        """
        self.data_frame = None
        self._matrix_column = list()

        self.profile_path = profiles_path
        self.profiles_filename = profiles_filename
        self.format_file = format_file

    @property
    def directories(self):
        """Property for getting the configuration of the directories.

        Returns:
            A dictionary with the path of the directory as key and a tuple
            with the postfix of the files and the column to save in the
            dataframe.

        """
        d_updated = {os.path.join(self.profile_path, k): v
                     for k, v in DIR_CONFIGURATION.items()}
        return d_updated

    def load(self):
        """Load the data into the attribute dataframe.

        Create a dataframe with the information of the main file and append
        new columns with the matrix saved in the files which are sorted by
        the configuration directories.

        Returns:
            None

        """
        index_list_error = list()

        if self.format_file == 'xls':
            path = os.path.join(self.profile_path, self.profiles_filename)

            self.data_frame = pandas.read_excel(path)
            self.data_frame.set_index(
                'ID', verify_integrity=True, inplace=True)

            for path, _, files in os.walk(self.profile_path):

                if files:
                    logging.info(
                        'Reading content in "{}" directory '
                        'with {} files'.format(path, len(files)))

                    try:
                        end_name, column_name = self.directories[path]
                    except KeyError:
                        continue
                    else:
                        if column_name not in self.data_frame.columns:
                            self.data_frame[column_name] = np.nan
                            self.data_frame[column_name] = \
                                self.data_frame[column_name].astype(np.ndarray)
                            self._matrix_column.append(column_name)

                        matrix_values = {
                            f: np.loadtxt(os.path.join(path, f),
                                          delimiter=",", skiprows=0, ndmin=2)
                            for f in filter(
                                lambda x: x.endswith(end_name), files)}

                        for file_name, matrix in matrix_values.items():
                            index_error = self._save_to_dataframe(
                                file_name, matrix, end_name, column_name)
                            if index_error is not None and \
                                    index_error not in index_list_error:
                                index_list_error.append(index_error)

            if index_list_error:
                logging.warning(
                    'Index {} couldn\'t be found '
                    'in main file (total {} index(es)).'.format(
                        ', '.join(index_list_error), len(index_list_error)
                    ))

        else:
            raise NotImplementedError(
                'Format \'{}\' not supported'.format(self.format_file))

    def _save_to_dataframe(self, file_name, matrix, postfix, column_name):

        index = file_name.replace(postfix, '')

        if index in self.data_frame.index:
            try:
                current_value = self.data_frame.at[index, column_name]
                if pandas.isnull(current_value) is True:
                    raise KeyError

            except KeyError:
                self.data_frame.at[index, column_name] = matrix
            else:
                raise ValueError(
                    'Already existing value at [{}, {}]: Value from {} '
                    'cannot be loaded.'.format(index, column_name, file_name))
        else:
            logging.debug('Index (id) {} couldn\'t be found, '
                          'skipped (file {})'.format(index, file_name))
            return index

    def binarize_matrix(self, columns, threshold=0, inplace=True):
        """Convert the columns with matrix in binary values.

        If the value is bigger than threshold 1 else 0

        Args:
            *columns: columns with the matrix to be binarized
            threshold: threshold for the matrix values, default 0
            inplace: if True overwrite the date frame attribute.

        Returns:
            A dataframe with binarized matrix.

        """
        df = binarize_matrix(self.data_frame, columns, threshold)

        if inplace:
            self.data_frame = df

        return df.copy()

    def combine_matrix(self, columns, column_result, func=np.mean,
                       inplace=True):
        """Apply a function to a set of columns with contains matrices.

        Args:
            columns: name of columns with matrix
            column_result: name of column for saving
            func: function to be applied
            inplace: if True overwrite the date frame attribute.

        Returns:
            A dataframe with a new column for the result.

        """
        df = combine_matrix(self.data_frame, columns, column_result, func)

        if inplace:
            self.data_frame = df

        return df.copy()

    def spread_out_matrix(self, columns, symmetric=True, keep_matrix=False,
                          inplace=True):
        """Spread out the values of matrix saved in columns.

        Args:
            *columns: columns with the matrix to be spread out
            symmetric: matrix are symmetric, ignored half matrix
            keep_matrix: if True the original column with the matrix wont be
            removed.
            inplace: if True overwrite the date frame attribute.

        Returns:
            A dataframe with a new column for each coordinate of the matrix and
            for each matrix.

        """
        df = spread_out_matrix(self.data_frame, columns, symmetric,
                               keep_matrix)

        if inplace:
            self.data_frame = df

        return df.copy()
