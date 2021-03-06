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
class TestExtedMatrix(object):

    def test_small(self, datafiles):
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

    def test_spread_out_simple(self, datafiles):
        path = os.path.join(str(datafiles), 'small_data')
        filename = 'profiles.xlsx'

        profiles = NeuProfiles(profiles_path=path, profiles_filename=filename)
        profiles.load()

        new_dataframe = profiles.spread_out_matrix(
            ['DTI_L1'], keep_matrix=False, symmetric=False)

        assert [
            'DTI_FA',
            'DTI_L1_0_0', 'DTI_L1_0_1', 'DTI_L1_0_2',
            'DTI_L1_1_0', 'DTI_L1_1_1', 'DTI_L1_1_2',
            'DTI_L1_2_0', 'DTI_L1_2_1', 'DTI_L1_2_2',
        ] == sorted(list(new_dataframe.columns))

        assert new_dataframe.DTI_L1_0_0[0] == 11
        assert new_dataframe.DTI_L1_0_1[0] == 12
        assert new_dataframe.DTI_L1_0_2[0] == 13

        assert new_dataframe.DTI_L1_1_0[0] == 14
        assert new_dataframe.DTI_L1_1_1[0] == 15
        assert new_dataframe.DTI_L1_1_2[0] == 16

        assert new_dataframe.DTI_L1_2_0[0] == 17
        assert new_dataframe.DTI_L1_2_1[0] == 18
        assert new_dataframe.DTI_L1_2_2[0] == 19

    def test_spread_out_multiple(self, datafiles):
        path = os.path.join(str(datafiles), 'small_data')
        filename = 'profiles.xlsx'

        profiles = NeuProfiles(profiles_path=path, profiles_filename=filename)
        profiles.load()

        new_dataframe = profiles.spread_out_matrix(['DTI_L1', 'DTI_FA'],
                                                   keep_matrix=False,
                                                   symmetric=False)

        assert [
            'DTI_FA_0_0', 'DTI_FA_0_1', 'DTI_FA_0_2',
            'DTI_FA_1_0', 'DTI_FA_1_1', 'DTI_FA_1_2',
            'DTI_FA_2_0', 'DTI_FA_2_1', 'DTI_FA_2_2',
            'DTI_L1_0_0', 'DTI_L1_0_1', 'DTI_L1_0_2',
            'DTI_L1_1_0', 'DTI_L1_1_1', 'DTI_L1_1_2',
            'DTI_L1_2_0', 'DTI_L1_2_1', 'DTI_L1_2_2',
        ] == sorted(list(new_dataframe.columns))

        assert new_dataframe.DTI_L1_0_0[0] == 11
        assert new_dataframe.DTI_L1_0_1[0] == 12
        assert new_dataframe.DTI_L1_0_2[0] == 13

        assert new_dataframe.DTI_L1_1_0[0] == 14
        assert new_dataframe.DTI_L1_1_1[0] == 15
        assert new_dataframe.DTI_L1_1_2[0] == 16

        assert new_dataframe.DTI_L1_2_0[0] == 17
        assert new_dataframe.DTI_L1_2_1[0] == 18
        assert new_dataframe.DTI_L1_2_2[0] == 19
        
        assert new_dataframe.DTI_FA_0_0[0] == 1
        assert new_dataframe.DTI_FA_0_1[0] == 2
        assert new_dataframe.DTI_FA_0_2[0] == 3

        assert new_dataframe.DTI_FA_1_0[0] == 4
        assert new_dataframe.DTI_FA_1_1[0] == 5
        assert new_dataframe.DTI_FA_1_2[0] == 6

        assert new_dataframe.DTI_FA_2_0[0] == 7
        assert new_dataframe.DTI_FA_2_1[0] == 8
        assert new_dataframe.DTI_FA_2_2[0] == 9

    def test_extract_multiple_keeping(self, datafiles):
        path = os.path.join(str(datafiles), 'small_data')
        filename = 'profiles.xlsx'

        profiles = NeuProfiles(profiles_path=path, profiles_filename=filename)
        profiles.load()

        new_dataframe = profiles.spread_out_matrix(['DTI_L1', 'DTI_FA',],
                                                   keep_matrix=True,
                                                   symmetric=False)

        assert [
            'DTI_FA',
            'DTI_FA_0_0', 'DTI_FA_0_1', 'DTI_FA_0_2',
            'DTI_FA_1_0', 'DTI_FA_1_1', 'DTI_FA_1_2',
            'DTI_FA_2_0', 'DTI_FA_2_1', 'DTI_FA_2_2',
            'DTI_L1',
            'DTI_L1_0_0', 'DTI_L1_0_1', 'DTI_L1_0_2',
            'DTI_L1_1_0', 'DTI_L1_1_1', 'DTI_L1_1_2',
            'DTI_L1_2_0', 'DTI_L1_2_1', 'DTI_L1_2_2',
        ] == sorted(list(new_dataframe.columns))

        assert new_dataframe.DTI_L1_0_0[0] == 11
        assert new_dataframe.DTI_L1_0_1[0] == 12
        assert new_dataframe.DTI_L1_0_2[0] == 13

        assert new_dataframe.DTI_L1_1_0[0] == 14
        assert new_dataframe.DTI_L1_1_1[0] == 15
        assert new_dataframe.DTI_L1_1_2[0] == 16

        assert new_dataframe.DTI_L1_2_0[0] == 17
        assert new_dataframe.DTI_L1_2_1[0] == 18
        assert new_dataframe.DTI_L1_2_2[0] == 19

        assert new_dataframe.DTI_FA_0_0[0] == 1
        assert new_dataframe.DTI_FA_0_1[0] == 2
        assert new_dataframe.DTI_FA_0_2[0] == 3

        assert new_dataframe.DTI_FA_1_0[0] == 4
        assert new_dataframe.DTI_FA_1_1[0] == 5
        assert new_dataframe.DTI_FA_1_2[0] == 6

        assert new_dataframe.DTI_FA_2_0[0] == 7
        assert new_dataframe.DTI_FA_2_1[0] == 8
        assert new_dataframe.DTI_FA_2_2[0] == 9

        assert profiles.data_frame.DTI_FA.FIS_007[0][0] == 1
        assert profiles.data_frame.DTI_FA.FIS_007[2][2] == 9

        assert profiles.data_frame.DTI_L1.FIS_007[0][0] == 11
        assert profiles.data_frame.DTI_L1.FIS_007[2][2] == 19

    def test_inplace_false(self, datafiles):
        path = os.path.join(str(datafiles), 'small_data')
        filename = 'profiles.xlsx'

        profiles = NeuProfiles(profiles_path=path, profiles_filename=filename)
        profiles.load()

        old_dataframe = profiles.data_frame

        new_dataframe = profiles.spread_out_matrix(
            ['DTI_L1'], keep_matrix=False, inplace=False, symmetric=False)

        assert [
            'DTI_FA',
            'DTI_L1_0_0', 'DTI_L1_0_1', 'DTI_L1_0_2',
            'DTI_L1_1_0', 'DTI_L1_1_1', 'DTI_L1_1_2',
            'DTI_L1_2_0', 'DTI_L1_2_1', 'DTI_L1_2_2',
        ] == sorted(list(new_dataframe.columns))

        for new_column in  [
            'DTI_L1_0_0', 'DTI_L1_0_1', 'DTI_L1_0_2',
            'DTI_L1_1_0', 'DTI_L1_1_1', 'DTI_L1_1_2',
            'DTI_L1_2_0', 'DTI_L1_2_1', 'DTI_L1_2_2',
        ]:
            assert new_column not in profiles.data_frame.columns

        assert (old_dataframe == profiles.data_frame).all().all()

    def test_inplace_true(self, datafiles):
        path = os.path.join(str(datafiles), 'small_data')
        filename = 'profiles.xlsx'

        profiles = NeuProfiles(profiles_path=path, profiles_filename=filename)
        profiles.load()

        new_dataframe = profiles.spread_out_matrix(
            ['DTI_L1'], keep_matrix=False, inplace=True, symmetric=False)

        assert [
            'DTI_FA',
            'DTI_L1_0_0', 'DTI_L1_0_1', 'DTI_L1_0_2',
            'DTI_L1_1_0', 'DTI_L1_1_1', 'DTI_L1_1_2',
            'DTI_L1_2_0', 'DTI_L1_2_1', 'DTI_L1_2_2',
        ] == sorted(list(new_dataframe.columns))

        for new_column in [
            'DTI_L1_0_0', 'DTI_L1_0_1', 'DTI_L1_0_2',
            'DTI_L1_1_0', 'DTI_L1_1_1', 'DTI_L1_1_2',
            'DTI_L1_2_0', 'DTI_L1_2_1', 'DTI_L1_2_2',
        ]:
            assert new_column in profiles.data_frame.columns

        assert (new_dataframe == profiles.data_frame).all().all()

    def test_symmetric_true(self, datafiles):

        path = os.path.join(str(datafiles), 'small_data')
        filename = 'profiles.xlsx'

        profiles = NeuProfiles(profiles_path=path, profiles_filename=filename)
        profiles.load()

        new_dataframe = profiles.spread_out_matrix(['DTI_L1', 'DTI_FA'],
                                                   keep_matrix=False,
                                                   symmetric=True)
        assert [
                   'DTI_FA_1_0',
                   'DTI_FA_2_0', 'DTI_FA_2_1',


                   'DTI_L1_1_0',
                   'DTI_L1_2_0', 'DTI_L1_2_1',
               ] == sorted(list(new_dataframe.columns))

        assert new_dataframe.DTI_L1_1_0[0] == 14

        assert new_dataframe.DTI_L1_2_0[0] == 17
        assert new_dataframe.DTI_L1_2_1[0] == 18

        assert new_dataframe.DTI_FA_1_0[0] == 4

        assert new_dataframe.DTI_FA_2_0[0] == 7
        assert new_dataframe.DTI_FA_2_1[0] == 8
