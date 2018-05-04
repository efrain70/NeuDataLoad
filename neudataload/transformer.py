from sklearn.base import BaseEstimator, TransformerMixin


from .profiles import NeuProfiles

from sklearn.decomposition import PCA, NMF

class NeoTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, param=None):
        pass

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X).transform(X)

    def transform(self, X, y=None):
        profiles = NeuProfiles(None, None)

        from pandas import DataFrame
        if isinstance(X, DataFrame):
            profiles.data_frame = X
        else:
            profiles.data_frame = DataFrame(data=X)

        with_matrix = ['DTI_FA', 'DTI_L1', 'DTI_MD', 'DTI_RX', ]  # 'RAW', 'FUNC', 'LS']


        # Combined [lista, nombre] o None
        profiles.combine_matrix(columns=with_matrix,
                                column_result='combined')

        # Spread [lista, keep] o None,
        profiles.spread_out_matrix(['combined', ], keep_matrix=False,
                                   inplace=True)

        # Qu'e columnas a considerar
        regex_columns = r'combined_*'  # |' + r'|'.join(columns)
        X_y = profiles.data_frame.filter(regex=(regex_columns))

        return X_y

