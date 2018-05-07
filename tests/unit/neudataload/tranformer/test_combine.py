import os

import numpy as np
import pytest
from sklearn.pipeline import Pipeline

from neudataload.profiles import NeuProfiles
from neudataload.transformer import CombineMatrixTransformer

FIXTURE_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    '..',
    '..',
    '..',
    'examples',
    )


@pytest.mark.datafiles(FIXTURE_DIR)
class TestCombinations(object):

    def test_pipe_simple(self, datafiles):

        path = os.path.join(str(datafiles), 'small_data')
        filename = 'profiles.xlsx'

        profiles = NeuProfiles(profiles_path=path, profiles_filename=filename)

        assert profiles.data_frame is None

        profiles.load()

        assert profiles.data_frame is not None

        pipe = Pipeline([
            ('combining', CombineMatrixTransformer(
                columns=['DTI_FA', 'DTI_L1'],
                column_name='combined')),
        ])
        output = pipe.fit_transform(profiles.data_frame)

        assert sorted(list(output.columns)) == ['DTI_FA', 'DTI_L1', 'combined']

        assert (output.combined.FIS_007 ==
                [[6, 7, 8, ],
                 [9, 10, 11],
                 [12, 13, 14]]).all()

    def test_pipe_params(self, datafiles):

        path = os.path.join(str(datafiles), 'small_data')
        filename = 'profiles.xlsx'

        profiles = NeuProfiles(profiles_path=path, profiles_filename=filename)

        assert profiles.data_frame is None

        profiles.load()

        assert profiles.data_frame is not None

        params = {'combining__columns': ['DTI_FA', 'DTI_L1'],
                  'combining__column_name': 'combined',
                  }

        pipe = Pipeline([
            ('combining', CombineMatrixTransformer()),
        ])
        pipe.set_params(**params)

        output = pipe.fit_transform(profiles.data_frame)

        assert sorted(list(output.columns)) == ['DTI_FA', 'DTI_L1', 'combined']

        assert (output.combined.FIS_007 ==
                [[6, 7, 8, ],
                 [9, 10, 11],
                 [12, 13, 14]]).all()

    def test_witout_name(self, datafiles):

        path = os.path.join(str(datafiles), 'small_data')
        filename = 'profiles.xlsx'

        profiles = NeuProfiles(profiles_path=path, profiles_filename=filename)

        assert profiles.data_frame is None

        profiles.load()

        assert profiles.data_frame is not None

        pipe = Pipeline([
            ('combining', CombineMatrixTransformer(
                columns=['DTI_FA', 'DTI_L1'])),
        ])
        output = pipe.fit_transform(profiles.data_frame)

        assert sorted(list(output.columns)) == ['DTI_FA', 'DTI_L1']

    def test_witout_columns(self, datafiles):

        path = os.path.join(str(datafiles), 'small_data')
        filename = 'profiles.xlsx'

        profiles = NeuProfiles(profiles_path=path, profiles_filename=filename)

        assert profiles.data_frame is None

        profiles.load()

        assert profiles.data_frame is not None

        pipe = Pipeline([
            ('combining', CombineMatrixTransformer(
                column_name='combined')),
        ])
        output = pipe.fit_transform(profiles.data_frame)

        assert sorted(list(output.columns)) == ['DTI_FA', 'DTI_L1']

    def test_no_df_simple(self, datafiles):

        path = os.path.join(str(datafiles), 'small_data')
        filename = 'profiles.xlsx'

        profiles = NeuProfiles(profiles_path=path, profiles_filename=filename)

        assert profiles.data_frame is None

        profiles.load()

        assert profiles.data_frame is not None

        pipe = Pipeline([
            ('combining', CombineMatrixTransformer(
                columns=['0', '1'],
                column_name='combined')),
        ])

        output = pipe.fit_transform(profiles.data_frame.values)

        assert (output[0][0] == np.asarray(
            [[11, 12, 13], [14, 15, 16], [17, 18, 19]])).all()

        assert (output[0][1] == np.asarray(
            [[1, 2, 3, ], [4, 5, 6, ], [7, 8, 9, ]])).all()

        assert (output[0][2] == np.asarray(
            [[6, 7, 8, ], [9, 10, 11], [12, 13, 14]])).all()
