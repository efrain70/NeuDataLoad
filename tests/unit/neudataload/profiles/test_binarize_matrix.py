import os

import pytest
import pandas

from neudataload.profiles import NeuProfiles


FIXTURE_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    '..',
    '..',
    '..',
    'examples',
    )


@pytest.mark.datafiles(FIXTURE_DIR)
class TestBinarizeMatrix(object):

    def test_simple(self, datafiles):
        path = os.path.join(str(datafiles), 'small_data')
        filename = 'profiles.xlsx'

        profiles = NeuProfiles(profiles_path=path, profiles_filename=filename)

        assert profiles.data_frame is None

        profiles.load()

        assert profiles.data_frame is not None
        assert isinstance(profiles.data_frame, pandas.DataFrame)
        assert profiles.data_frame.shape == (1, 2)
        assert profiles.data_frame.index.name == 'ID'
        assert sorted(
            list(profiles.data_frame.columns)) == ['DTI_FA', 'DTI_L1']

        assert profiles.data_frame.DTI_FA.FIS_007[0][0] == 1
        assert profiles.data_frame.DTI_FA.FIS_007[2][2] == 9

        assert profiles.data_frame.DTI_L1.FIS_007[0][0] == 11
        assert profiles.data_frame.DTI_L1.FIS_007[2][2] == 19

        assert profiles.data_frame.DTI_FA.FIS_007[0][0] == 1
        assert profiles.data_frame.DTI_FA.FIS_007[2][2] == 9

        assert profiles.data_frame.DTI_L1.FIS_007[0][0] == 11
        assert profiles.data_frame.DTI_L1.FIS_007[2][2] == 19

        profiles.binarize_matrix(['DTI_FA', 'DTI_L1'])

        assert profiles.data_frame.DTI_FA.FIS_007[0][0] == 1
        assert profiles.data_frame.DTI_FA.FIS_007[2][2] == 1

        assert profiles.data_frame.DTI_L1.FIS_007[0][0] == 1
        assert profiles.data_frame.DTI_L1.FIS_007[2][2] == 1

        assert profiles.data_frame.DTI_FA.FIS_007[0][0] == 1
        assert profiles.data_frame.DTI_FA.FIS_007[2][2] == 1

        assert profiles.data_frame.DTI_L1.FIS_007[0][0] == 1
        assert profiles.data_frame.DTI_L1.FIS_007[2][2] == 1

    def test_threshold(self, datafiles):
        path = os.path.join(str(datafiles), 'small_data')
        filename = 'profiles.xlsx'

        profiles = NeuProfiles(profiles_path=path, profiles_filename=filename)

        assert profiles.data_frame is None

        profiles.load()

        assert profiles.data_frame is not None
        assert isinstance(profiles.data_frame, pandas.DataFrame)
        assert profiles.data_frame.shape == (1, 2)
        assert profiles.data_frame.index.name == 'ID'
        assert sorted(
            list(profiles.data_frame.columns)) == ['DTI_FA', 'DTI_L1']

        assert profiles.data_frame.DTI_FA.FIS_007[0][0] == 1
        assert profiles.data_frame.DTI_FA.FIS_007[2][2] == 9

        assert profiles.data_frame.DTI_L1.FIS_007[0][0] == 11
        assert profiles.data_frame.DTI_L1.FIS_007[2][2] == 19

        assert profiles.data_frame.DTI_FA.FIS_007[0][0] == 1
        assert profiles.data_frame.DTI_FA.FIS_007[2][2] == 9

        assert profiles.data_frame.DTI_L1.FIS_007[0][0] == 11
        assert profiles.data_frame.DTI_L1.FIS_007[2][2] == 19

        profiles.binarize_matrix(['DTI_FA', 'DTI_L1'], threshold=1)

        assert profiles.data_frame.DTI_FA.FIS_007[0][0] == 0
        assert profiles.data_frame.DTI_FA.FIS_007[2][2] == 1

        assert profiles.data_frame.DTI_L1.FIS_007[0][0] == 1
        assert profiles.data_frame.DTI_L1.FIS_007[2][2] == 1

        assert profiles.data_frame.DTI_FA.FIS_007[0][0] == 0
        assert profiles.data_frame.DTI_FA.FIS_007[2][2] == 1

        assert profiles.data_frame.DTI_L1.FIS_007[0][0] == 1
        assert profiles.data_frame.DTI_L1.FIS_007[2][2] == 1
