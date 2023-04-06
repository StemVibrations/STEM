from stem import stem


def test_1(capsys):
    stem.main()
    captured = capsys.readouterr()
    assert captured.out == "STEM is running\n"
