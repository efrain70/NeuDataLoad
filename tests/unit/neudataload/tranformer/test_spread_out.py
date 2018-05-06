import os

import numpy as np
import pytest
from sklearn.pipeline import Pipeline

from neudataload.profiles import NeuProfiles
from neudataload.transformer import SpreadOutMatrixTransformer

FIXTURE_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    '..',
    '..',
    '..',
    'examples',
    )


@pytest.mark.datafiles(FIXTURE_DIR)
class TestSpreadout(object):

    def test_pipe_all(self, datafiles):

        path = os.path.join(str(datafiles), 'small_data')
        filename = 'profiles.xlsx'

        profiles = NeuProfiles(profiles_path=path, profiles_filename=filename)

        assert profiles.data_frame is None

        profiles.load()

        assert profiles.data_frame is not None

        pipe = Pipeline([
            ('spread', SpreadOutMatrixTransformer(
                columns=profiles.data_frame.columns,)
             ),
        ])
        output = pipe.fit_transform(profiles.data_frame)

        assert sorted(list(output.columns)) == [
             'DTI_FA_1_0', 'DTI_FA_2_0', 'DTI_FA_2_1',
             'DTI_L1_1_0', 'DTI_L1_2_0', 'DTI_L1_2_1']

        assert output.DTI_FA_1_0.FIS_007 == 4
        assert output.DTI_FA_2_0.FIS_007 == 7
        assert output.DTI_FA_2_1.FIS_007 == 8
        assert output.DTI_L1_1_0.FIS_007 == 14
        assert output.DTI_L1_2_0.FIS_007 == 17
        assert output.DTI_L1_2_1.FIS_007 == 18

    def test_pipe_default(self, datafiles):

        path = os.path.join(str(datafiles), 'small_data')
        filename = 'profiles.xlsx'

        profiles = NeuProfiles(profiles_path=path, profiles_filename=filename)

        assert profiles.data_frame is None

        profiles.load()

        assert profiles.data_frame is not None

        pipe = Pipeline([
            ('selecting', SpreadOutMatrixTransformer(
                columns=None)
             ),
        ])
        output = pipe.fit_transform(profiles.data_frame)

        assert sorted(list(output.columns)) == ['DTI_FA', 'DTI_L1', ]
        assert (output == profiles.data_frame).all().all()

    def test_pipe_symmetric(self, datafiles):
        path = os.path.join(str(datafiles), 'small_data')
        filename = 'profiles.xlsx'

        profiles = NeuProfiles(profiles_path=path, profiles_filename=filename)

        assert profiles.data_frame is None

        profiles.load()

        assert profiles.data_frame is not None

        pipe = Pipeline([
            ('spread', SpreadOutMatrixTransformer(
                columns=profiles.data_frame.columns, symmetric=False)
             ),
        ])
        output = pipe.fit_transform(profiles.data_frame)
        assert sorted(list(output.columns)) == [
             'DTI_FA_0_0', 'DTI_FA_0_1', 'DTI_FA_0_2',
             'DTI_FA_1_0', 'DTI_FA_1_1', 'DTI_FA_1_2',
             'DTI_FA_2_0', 'DTI_FA_2_1', 'DTI_FA_2_2',

             'DTI_L1_0_0', 'DTI_L1_0_1', 'DTI_L1_0_2',
             'DTI_L1_1_0', 'DTI_L1_1_1', 'DTI_L1_1_2',
             'DTI_L1_2_0', 'DTI_L1_2_1', 'DTI_L1_2_2']

        assert output.DTI_FA_0_0.FIS_007 == 1
        assert output.DTI_FA_0_1.FIS_007 == 2
        assert output.DTI_FA_0_2.FIS_007 == 3
        assert output.DTI_FA_1_0.FIS_007 == 4
        assert output.DTI_FA_1_1.FIS_007 == 5
        assert output.DTI_FA_1_2.FIS_007 == 6
        assert output.DTI_FA_2_0.FIS_007 == 7
        assert output.DTI_FA_2_1.FIS_007 == 8
        assert output.DTI_FA_2_2.FIS_007 == 9

        assert output.DTI_L1_0_0.FIS_007 == 11
        assert output.DTI_L1_0_1.FIS_007 == 12
        assert output.DTI_L1_0_2.FIS_007 == 13
        assert output.DTI_L1_1_0.FIS_007 == 14
        assert output.DTI_L1_1_1.FIS_007 == 15
        assert output.DTI_L1_1_2.FIS_007 == 16
        assert output.DTI_L1_2_0.FIS_007 == 17
        assert output.DTI_L1_2_1.FIS_007 == 18
        assert output.DTI_L1_2_2.FIS_007 == 19

    def test_no_df_simple(self, datafiles):

        path = os.path.join(str(datafiles), 'small_data')
        filename = 'profiles.xlsx'

        profiles = NeuProfiles(profiles_path=path, profiles_filename=filename)

        assert profiles.data_frame is None

        profiles.load()

        assert profiles.data_frame is not None

        pipe = Pipeline([
            ('spread', SpreadOutMatrixTransformer(columns=[0, 1])),
        ])
        output = pipe.fit_transform(profiles.data_frame.values)

        assert (output == np.array([[14., 17., 18.,  4.,  7.,  8.]])).all()
