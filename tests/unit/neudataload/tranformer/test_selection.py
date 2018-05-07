import os

import pytest
from sklearn.pipeline import Pipeline

from neudataload.profiles import NeuProfiles
from neudataload.transformer import (
    FeatureMatrixTransformer,
    SpreadOutMatrixTransformer
)

FIXTURE_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    '..',
    '..',
    '..',
    'examples',
    )


@pytest.mark.datafiles(FIXTURE_DIR)
class TestSelection(object):

    def test_pipe_default(self, datafiles):

        path = os.path.join(str(datafiles), 'small_data')
        filename = 'profiles.xlsx'

        profiles = NeuProfiles(profiles_path=path, profiles_filename=filename)

        assert profiles.data_frame is None

        profiles.load()

        assert profiles.data_frame is not None

        pipe = Pipeline([
            ('selecting', FeatureMatrixTransformer(
                columns=profiles.data_frame.columns,)
             ),
        ])
        output = pipe.fit_transform(profiles.data_frame)

        assert sorted(list(output.columns)) == ['DTI_FA', 'DTI_L1', ]
        assert (output == profiles.data_frame).all().all()

    def test_pipe_simple(self, datafiles):

        path = os.path.join(str(datafiles), 'small_data')
        filename = 'profiles.xlsx'

        profiles = NeuProfiles(profiles_path=path, profiles_filename=filename)

        assert profiles.data_frame is None

        profiles.load()

        assert profiles.data_frame is not None

        pipe = Pipeline([
            ('selecting', FeatureMatrixTransformer(
                columns=None, matrix_columns=None)
             ),
        ])
        output = pipe.fit_transform(profiles.data_frame)

        assert sorted(list(output.columns)) == ['DTI_FA', 'DTI_L1', ]
        assert (output == profiles.data_frame).all().all()

    def test_pipe_column(self, datafiles):

        path = os.path.join(str(datafiles), 'small_data')
        filename = 'profiles.xlsx'

        profiles = NeuProfiles(profiles_path=path, profiles_filename=filename)

        assert profiles.data_frame is None

        profiles.load()

        assert profiles.data_frame is not None

        pipe = Pipeline([
            ('selecting', FeatureMatrixTransformer(
                columns=['DTI_FA'], matrix_columns=None)
             ),
        ])
        output = pipe.fit_transform(profiles.data_frame)

        assert sorted(list(output.columns)) == ['DTI_FA', ]
        assert (output.DTI_FA == profiles.data_frame.DTI_FA).all()

    def test_pipe_spread_out_column(self, datafiles):

        path = os.path.join(str(datafiles), 'small_data')
        filename = 'profiles.xlsx'

        profiles = NeuProfiles(profiles_path=path, profiles_filename=filename)

        assert profiles.data_frame is None

        profiles.load()

        assert profiles.data_frame is not None

        pipe = Pipeline([
            ('spread', SpreadOutMatrixTransformer(columns=['DTI_FA',
                                                           'DTI_L1'])),
            ('selecting', FeatureMatrixTransformer(
                matrix_columns=['DTI_FA'], columns=None)),
        ])
        output = pipe.fit_transform(profiles.data_frame)

        assert sorted(list(output.columns)) == [
            'DTI_FA_1_0', 'DTI_FA_2_0', 'DTI_FA_2_1']

        assert output.DTI_FA_1_0.FIS_007 == 4
        assert output.DTI_FA_2_0.FIS_007 == 7
        assert output.DTI_FA_2_1.FIS_007 == 8

    def test_no_df_simple(self, datafiles):

        path = os.path.join(str(datafiles), 'small_data')
        filename = 'profiles.xlsx'

        profiles = NeuProfiles(profiles_path=path, profiles_filename=filename)

        assert profiles.data_frame is None

        profiles.load()

        assert profiles.data_frame is not None

        pipe = Pipeline([
            ('spread', SpreadOutMatrixTransformer(columns=['0', '1'])),
            ('selecting', FeatureMatrixTransformer(
                matrix_columns=['0'], columns=None)),
        ])
        output = pipe.fit_transform(profiles.data_frame.values)

        assert int(output[0]) == 14
