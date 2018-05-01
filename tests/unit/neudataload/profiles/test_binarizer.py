import os

import pytest

from neudataload.profiles import NeuProfiles


FIXTURE_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    '..',
    '..',
    '..',
    'examples',
    )


@pytest.mark.datafiles(FIXTURE_DIR)
class TestBinarize(object):

    def test_simple(self, datafiles):
        path = os.path.join(str(datafiles), 'valid')
        filename = 'profiles.xlsx'

        profiles = NeuProfiles(profiles_path=path, profiles_filename=filename)

        assert profiles.data_frame is None

        profiles.load()

        groups = {
          'A': [1],
          'B': [2],
          'C': [3],
          'A-B': [1, 2],
          'A-B-C': [1, 2, 3],
          'X': [],
          'W': []
        }
        values = profiles.get_multilabel('profile', groups)

        # 3 columns, 63 items
        assert values.shape == (63, 3)

        assert (values[profiles.data_frame.profile == 'A'] == [1, 0, 0]).all()
        assert (values[profiles.data_frame.profile == 'B'] == [0, 1, 0]).all()
        assert (values[profiles.data_frame.profile == 'C'] == [0, 0, 1]).all()
        assert (values[profiles.data_frame.profile == 'C'] == [0, 0, 1]).all()
        assert (values[profiles.data_frame.profile == 'A-B'] == [1, 1, 0]).all()

        assert (values[profiles.data_frame.profile == 'A-B-C'] == [1, 1, 1]).all()

        assert (values[profiles.data_frame.profile == 'X'] == [0, 0, 0]).all()

    def test_not_found(self, datafiles):
        path = os.path.join(str(datafiles), 'valid')
        filename = 'profiles.xlsx'

        profiles = NeuProfiles(profiles_path=path, profiles_filename=filename)

        assert profiles.data_frame is None

        profiles.load()

        groups = {
          'A': [1],
          'B': [2],
          'C': [3],
          'A-B': [1, 2],
          'A-B-C': [1, 2, 3],
          'X': [],
        }
        with pytest.raises(KeyError) as error:
            profiles.get_multilabel('profile', groups)

        # W not found
        assert str(error.value) == 'W'
