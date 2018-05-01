import os
import logging

import pytest
import pandas
import six

from neudataload.profiles import NeuProfiles


FIXTURE_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    '..',
    '..',
    '..',
    'examples',
    )


@pytest.mark.datafiles(FIXTURE_DIR)
class TestLoad(object):

    def test_valid(self, datafiles):
        path = os.path.join(str(datafiles), 'valid')
        filename = 'profiles.xlsx'

        profiles = NeuProfiles(profiles_path=path, profiles_filename=filename)

        assert profiles.data_frame is None

        profiles.load()

        assert profiles.data_frame is not None
        assert isinstance(profiles.data_frame, pandas.DataFrame)
        assert profiles.data_frame.shape == (63, 8)
        assert profiles.data_frame.index.name == 'ID'

        assert 'profile' == profiles.data_frame.columns[0]

        for column_name in ['RAW', 'LS', 'DTI_L1', 'DTI_MD',
                            'DTI_RX', 'DTI_FA', 'FUNC']:
            assert column_name in profiles.data_frame.columns
            first_i = profiles.data_frame[column_name].first_valid_index()
            assert profiles.data_frame[column_name][first_i].shape == (76, 76)

    def test_fail_duplicates_id(self, datafiles):
        path = os.path.join(str(datafiles), 'duplicated_id')
        filename = 'profiles.xlsx'
        profiles = NeuProfiles(profiles_path=path, profiles_filename=filename)

        with pytest.raises(ValueError) as error:
            profiles.load()

        if six.PY2:
            assert str(error.value) == "Index has duplicate keys: [u'FIS_007', u'TTO_06']"
        else:
            assert str(error.value) == "Index has duplicate keys: ['FIS_007', 'TTO_06']"

    def test_fail_duplicates_data(self, datafiles):
        path = os.path.join(str(datafiles), 'duplicated_data')
        filename = 'profiles.xlsx'
        profiles = NeuProfiles(profiles_path=path, profiles_filename=filename)

        with pytest.raises(ValueError) as error:
            profiles.load()

        assert str(error.value) == "Already existing value at [FIS_007, DTI_FA]: " \
                                   "Value from FIS_007_FA_factor.csv cannot be loaded."

    def test_fail_format(self, datafiles):
        path = os.path.join(str(datafiles), 'duplicated_data')
        filename = 'profiles.xlsx'
        profiles = NeuProfiles(profiles_path=path, profiles_filename=filename, format_file='mat')

        with pytest.raises(NotImplementedError) as error:
            profiles.load()

        assert str(error.value) == 'Format \'mat\' not supported'

    def test_format_info(self, datafiles, caplog):
        caplog.set_level(logging.INFO)

        path = os.path.join(str(datafiles), 'valid')
        filename = 'profiles.xlsx'

        profiles = NeuProfiles(profiles_path=path, profiles_filename=filename)

        assert profiles.data_frame is None

        profiles.load()

        assert len(caplog.records) == 25

        for record in caplog.records:
            assert record.levelname == 'INFO'
            assert 'Reading content in ' in record.message
            assert path in record.message
            assert 'directory with ' in record.message

    def test_debug(self, datafiles, caplog):
        caplog.set_level(logging.INFO)
        path = os.path.join(str(datafiles), 'index_not_found')
        filename = 'profiles.xlsx'
        profiles = NeuProfiles(profiles_path=path, profiles_filename=filename)
        profiles.load()
        assert len(caplog.records) == 4

        assert caplog.records[2].levelname == 'WARNING'
        assert caplog.records[2].message == "Index (id) FIS_007 couldn't be found, skipped " \
                                            "(file FIS_007_FA_factor.csv)"
