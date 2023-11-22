import pytest

from stem.table import Table


class TestTable:

    def test_correct_table_creation(self):
        """
        Tests that table is created correctly
        """
        time = [0, 1, 2, 3, 4, 5]
        values = [0, 5, 10, 5, 0, 0]

        # correct definition of a table
        Table(times=time, values=values)

        # mismatch between times and values
        values = [0, 5, 10, 5, 0, 0, 10, 2]
                       
        msg1 = ("Dimension mismatch between times and values in table:\n"
                 f" - times: {len(time)}\n"
                 f" - values: {len(values)}\n")

        with pytest.raises(ValueError, match=msg1):
            Table(times=time, values=values)
