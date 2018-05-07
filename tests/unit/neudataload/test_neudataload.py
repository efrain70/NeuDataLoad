class TestNeuDataLoad(object):

    def test_import_version(self):
        from neudataload import __version__
        assert __version__ is not None

    def test_import_profiles(self):
        from neudataload import NeuProfiles
        assert NeuProfiles is not None

    def test_import_utils(self):
        from neudataload import spread_out_matrix
        assert spread_out_matrix is not None

        from neudataload import combine_matrix
        assert combine_matrix is not None

        from neudataload import binarize_matrix
        assert binarize_matrix is not None

        from neudataload import get_multilabel
        assert get_multilabel is not None
