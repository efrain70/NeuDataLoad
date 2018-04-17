from neudataload.profiles import NeuProfiles


class TestInitNeuDataLoad(object):
    """
    """

    def test_default(self):
        profiles = NeuProfiles('path', 'filename')

        assert profiles.data_frame is None
        assert profiles.profile_path == 'path'
        assert profiles.profiles_filename == 'filename'
        assert profiles.format_file == 'xls'

    def test_parameters(self):
        profiles = NeuProfiles(profiles_path='path', format_file='xls2', profiles_filename='my_file.xls')

        assert profiles.data_frame is None
        assert profiles.profile_path == 'path'
        assert profiles.profiles_filename == 'my_file.xls'
        assert profiles.format_file == 'xls2'
