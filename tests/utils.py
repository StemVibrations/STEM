import pprint
from typing import Dict, Any

import numpy.testing as npt


class TestUtils:

    @staticmethod
    def assert_dictionary_almost_equal(expected: Dict[Any, Any], actual: Dict[Any, Any]):
        """
        Checks whether two dictionaries are equal.

        Args:
            expected: Expected dictionary.
            actual: Actual dictionary.

        """

        for k, v in expected.items():

            assert k in actual

            if isinstance(v, dict):
                TestUtils.assert_dictionary_almost_equal(v, actual[k])
            elif isinstance(v, str):
                try:
                    assert v == actual[k]
                except AssertionError as me:
                    print('Expected:\n')
                    pprint.pprint(v)
                    print('Actual:\n')
                    pprint.pprint(v)
                    raise(AssertionError, me)
            elif isinstance(v, list):
                assert len(v) == len(actual[k])
                for v_i, actual_i in zip(v, actual[k]):
                    if isinstance(v_i, dict):
                        TestUtils.assert_dictionary_almost_equal(v_i, actual_i)
                    elif isinstance(v_i, str):
                        assert v_i == actual_i
                    else:
                        npt.assert_allclose(v_i, actual_i)

            else:
                npt.assert_allclose(v, actual[k])