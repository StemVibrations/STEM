import pytest

from stem.table import Table


class TestTable:

    def test_correct_table_creation(self):
        """
        Tests that table is created correctly
        """
        _time = [0, 1, 2, 3, 4, 5]
        _value1 = [0, 5, 10, 5, 0, 0]

        Table(times=_time, values=_value1)

        _value2 = [0, 5, 10, 5, 0, 0, 10, 2]
                       
        _msg1 = ("Dimension mismatch between times and values in table:\n"
                 f" - times: {len(_time)}\n"
                 f" - values: {len(_value2)}\n")
        with pytest.raises(ValueError, match=_msg1):
            Table(times=_time, values=_value2)
        