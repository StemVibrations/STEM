import pytest

from stem.table import Table


class TestTable:

    def test_correct_table_creation(self):
        """
        Tests that table is created correctly
        """
        _time = [0, 1, 2, 3, 4, 5]
        _amplitude1 = [0, 5, 10, 5, 0, 0]

        Table(step=_time, amplitude=_amplitude1, name="Table_1")

    def test_table_validation(self):
        """
        Tests that table raises correct errors when validating inputs.
        """
        _time = [0, 1, 2, 3, 4, 5]
        _amplitude1 = [0, 5, 10, 5, 0, 0]
        _amplitude2 = [0, 5, 10, 5, 0, 0, 10, 2]

        _msg1 = "Specified step is not understood: steps.\nPlease specify one `step` or `time` for table: Table_1"
        with pytest.raises(ValueError, match=_msg1):
            Table(step=_time, amplitude=_amplitude1, name="Table_1", step_type="steps")

        _msg2 = "Dimension mismatch between time/step and amplitudes in table: Table_2"
        with pytest.raises(ValueError, match=_msg2):
            Table(step=_time, amplitude=_amplitude2, name="Table_2")

        _msg3 = "id attribute should not be specified by the user in table: Table_3"
        with pytest.raises(ValueError, match=_msg3):
            Table(step=_time, amplitude=_amplitude1, name="Table_3", id=999)
