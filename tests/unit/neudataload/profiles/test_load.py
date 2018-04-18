import pytest
import pandas
import six

from neudataload.profiles import NeuProfiles

import os


FIXTURE_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    '..',
    '..',
    '..',
    'examples',
    )


class TestLoad(object):

    @pytest.mark.datafiles(FIXTURE_DIR)
    def test_load(self, datafiles):
        path = os.path.join(str(datafiles), 'valid')
        filename = 'profiles.xlsx'

        profiles = NeuProfiles(profiles_path=path, profiles_filename=filename)

        assert profiles.data_frame is None

        profiles.load()

        assert profiles.data_frame is not None
        assert isinstance(profiles.data_frame, pandas.DataFrame)
        assert profiles.data_frame.shape == (63, 7)
        assert profiles.data_frame.index.name == 'ID'

        for column_name in ['RAW', 'LS', 'DTI_L1', 'DTI_MD', 'DTI_RX', 'DTI_FA', 'FUNC']:
            assert column_name in profiles.data_frame.columns

    @pytest.mark.datafiles(FIXTURE_DIR)
    def test_load_fail_duplicates_id(self, datafiles):
        path = os.path.join(str(datafiles), 'duplicated_id')
        filename = 'profiles.xlsx'
        profiles = NeuProfiles(profiles_path=path, profiles_filename=filename)

        with pytest.raises(ValueError) as error:
            profiles.load()

        if six.PY2:
            assert str(error.value) == "Index has duplicate keys: [u'FIS_007', u'TTO_06']"
        else:
            assert str(error.value) == "Index has duplicate keys: ['FIS_007', 'TTO_06']"

    @pytest.mark.datafiles(FIXTURE_DIR)
    def test_load_fail_duplicates_data(self, datafiles):
        path = os.path.join(str(datafiles), 'duplicated_data')
        filename = 'profiles.xlsx'
        profiles = NeuProfiles(profiles_path=path, profiles_filename=filename)

        # TODO change exception
        with pytest.raises(Exception) as error:
            profiles.load()

        assert str(error.value) == "Already existing value at [FIS_007, DTI_FA]: " \
                                   "Value from FIS_007_FA_factor.csv cannot be loaded."

    @pytest.mark.datafiles(FIXTURE_DIR)
    def test_load_fail_format(self, datafiles):
        path = os.path.join(str(datafiles), 'duplicated_data')
        filename = 'profiles.xlsx'
        profiles = NeuProfiles(profiles_path=path, profiles_filename=filename, format_file='mat')

        # TODO change exception
        with pytest.raises(NotImplementedError) as error:
            profiles.load()

        # TODO add comment
        assert str(error.value) == ""
