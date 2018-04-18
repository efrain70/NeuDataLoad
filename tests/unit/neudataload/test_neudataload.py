class TestNeuDataLoad(object):

    def test_import_version(self):
        from neudataload import __version__
        assert __version__ is not None

    def test_import_profiles(self):
        from neudataload import NeuProfiles
        assert NeuProfiles is not None
