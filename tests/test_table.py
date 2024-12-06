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

    def test_interpolate_value_at_time(self):
        """
        Tests that the correct value is interpolated at a given time
        """
        time = [1.0, 2.0, 3.0]
        values = [1.0, 5.0, 12.0]

        table = Table(times=time, values=values)

        # test time before the first time
        assert table.interpolate_value_at_time(0) == 1.0

        # test for time which is part of the table
        assert table.interpolate_value_at_time(1) == 1.0

        # test interpolation
        assert table.interpolate_value_at_time(2.5) == 8.5

        # test for time after the last time
        assert table.interpolate_value_at_time(5) == 12.0
