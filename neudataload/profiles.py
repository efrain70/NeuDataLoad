import os

import numpy as np
import pandas

DIR_CONFIGURATION = {
    # "CONTROLS",
    # "CONTROLS/ADJACENCY_MATRICES_GRAPH",
    # "CONTROLS/ADJACENCY_MATRICES_GRAPH/DTI_indices",

    "CONTROLS/ADJACENCY_MATRICES_GRAPH/DTI_indices/FA": ('_FA_factor.csv', 'DTI_FA'),
    "CONTROLS/ADJACENCY_MATRICES_GRAPH/DTI_indices/L1": ('_L1_factor.csv', 'DTI_L1'),
    "CONTROLS/ADJACENCY_MATRICES_GRAPH/DTI_indices/MD": ('_MD_factor.csv', 'DTI_MD'),
    "CONTROLS/ADJACENCY_MATRICES_GRAPH/DTI_indices/RX": ('_RX_factor.csv', 'DTI_RX'),
    "CONTROLS/ADJACENCY_MATRICES_GRAPH/RAW": ('_weight_factor.csv', 'RAW'),

    # "FIS",
    # "FIS/ADJACENCY_MATRICES_GRAPH",
    # "FIS/ADJACENCY_MATRICES_GRAPH/DTI_indices",

    "FIS/ADJACENCY_MATRICES_GRAPH/DTI_indices/FA": ('_FA_factor.csv', 'DTI_FA'),
    "FIS/ADJACENCY_MATRICES_GRAPH/DTI_indices/L1": ('_L1_factor.csv', 'DTI_L1'),
    "FIS/ADJACENCY_MATRICES_GRAPH/DTI_indices/MD": ('_MD_factor.csv', 'DTI_MD'),
    "FIS/ADJACENCY_MATRICES_GRAPH/DTI_indices/RX": ('_RX_factor.csv', 'DTI_RX'),

    "FIS/ADJACENCY_MATRICES_GRAPH/FUNC": ('_r_matrix.csv', 'FUNC'),
    "FIS/ADJACENCY_MATRICES_GRAPH/LS": ('_matrix_LS.csv', 'LS'),
    "FIS/ADJACENCY_MATRICES_GRAPH/RAW": ('_weight_factor.csv', 'RAW'),

    # "PREDICTORS",
    # "PREDICTORS/ADJACENCY_MATRICES_GRAPH",
    # "PREDICTORS/ADJACENCY_MATRICES_GRAPH/DTI_indices",

    "PREDICTORS/ADJACENCY_MATRICES_GRAPH/DTI_indices/FA.csv": ('_FA_factor', 'DTI_FA'),
    "PREDICTORS/ADJACENCY_MATRICES_GRAPH/DTI_indices/L1.csv": ('_L1_factor', 'DTI_L1'),
    "PREDICTORS/ADJACENCY_MATRICES_GRAPH/DTI_indices/MD.csv": ('_MD_factor', 'DTI_MD'),
    "PREDICTORS/ADJACENCY_MATRICES_GRAPH/DTI_indices/RX.csv": ('_RX_factor', 'DTI_RX'),

    "PREDICTORS/ADJACENCY_MATRICES_GRAPH/LS": ('_matrix_LS.csv', 'LS'),
    "PREDICTORS/ADJACENCY_MATRICES_GRAPH/RAW": ('_weight_factor.csv', 'RAW'),

    # "REESCAN",
    # "REESCAN/ADJACENCY_MATRICES_GRAPH",
    # "REESCAN/ADJACENCY_MATRICES_GRAPH/DTI_indices",

    "REESCAN/ADJACENCY_MATRICES_GRAPH/DTI_indices/FA": ('_FA_factor.csv', 'DTI_FA'),
    "REESCAN/ADJACENCY_MATRICES_GRAPH/DTI_indices/L1": ('_L1_factor.csv', 'DTI_L1'),
    "REESCAN/ADJACENCY_MATRICES_GRAPH/DTI_indices/MD": ('_MD_factor.csv', 'DTI_MD'),
    "REESCAN/ADJACENCY_MATRICES_GRAPH/DTI_indices/RX": ('_RX_factor.csv', 'DTI_RX'),

    "REESCAN/ADJACENCY_MATRICES_GRAPH/LS": ('_matrix_LS.csv', 'LS'),
    "REESCAN/ADJACENCY_MATRICES_GRAPH/RAW": ('_weight_factor.csv', 'RAW'),
}


class NeuProfiles(object):
    """Main class for loading profiles and matrix in sub directories."""

    def __init__(self, profiles_path, profiles_filename, format_file='xls'):
        """
        """
        self.data_frame = None

        self.profile_path = profiles_path
        self.profiles_filename = profiles_filename
        self.format_file = format_file

    @property
    def directories(self):
        
        d_updated = {os.path.join(self.profile_path, k): v for k, v in DIR_CONFIGURATION.items()}
        return d_updated

    def load(self):

        if self.format_file == 'xls':
            path = os.path.join(self.profile_path, self.profiles_filename)

            self.data_frame = pandas.read_excel(path)
            self.data_frame.set_index('ID', verify_integrity=True, inplace=True)

            for path, _, files in os.walk(self.profile_path):
                # TODO logger
                # print(path)
                try:
                    end_name, column_name = self.directories[path]
                except KeyError:
                    continue
                else:
                    if column_name not in self.data_frame.columns:
                        self.data_frame[column_name] = np.nan
                        self.data_frame[column_name] = self.data_frame[column_name].astype(np.ndarray)

                    matrix_values = {f: np.loadtxt(os.path.join(path, f), delimiter=",", skiprows=0)
                                     for f in filter(lambda x: x.endswith(end_name), files)}

                    for file_name, matrix in matrix_values.items():
                        self._save_to_dataframe(file_name, matrix, end_name, column_name)

        else:
            # TODO comment
            raise NotImplemented

    def _save_to_dataframe(self, file_name, matrix, postfix, column_name):
            index = file_name.replace(postfix, '')

            if index in self.data_frame.index:
                try:
                    current_value = self.data_frame.at[index, column_name]
                    if pandas.isna(current_value) or not current_value:
                        raise KeyError

                except KeyError:
                    self.data_frame.at[index, column_name] = matrix
                else:
                    # TODO Change exception
                    raise Exception(
                        'Duplicated key-column value {} {} from {}: current {}'.format(
                            index, column_name, file_name, str(current_value)))
            else:
                # TODO logger
                pass