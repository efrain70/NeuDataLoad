from neudataload.utils import all_combinations


class TestCombinations(object):

    def test_simple_list(self):
        all_items = all_combinations([0, 1, 2, ])
        assert list(all_items) == [(0,), (1,), (2,),
                                   (0, 1), (0, 2),  (1, 2),
                                   (0, 1, 2)]

    def test_simple_tuple(self):
        all_items = all_combinations((0, 1, 2, ))
        assert list(all_items) == [(0,), (1,), (2,),
                                   (0, 1), (0, 2),  (1, 2),
                                   (0, 1, 2)]

    def test_empty(self):
        all_items = all_combinations([])
        assert list(all_items) == list()

