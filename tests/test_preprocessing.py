from aphrodite.preprocessing import Preprocessor


def test_tokenize_removes_special_chars():
    pre = Preprocessor(remove_stopwords=False, stem=False)
    assert pre.process("Hello, world! C++ & Python3.") == [
        "hello",
        "world",
        "c",
        "python3",
    ]


def test_lowercasing():
    pre = Preprocessor(remove_stopwords=False, stem=False)
    assert pre.process("MACHINE Learning") == ["machine", "learning"]


def test_stopwords_removed():
    pre = Preprocessor(stem=False)
    tokens = pre.process("the quick brown fox and the lazy dog")
    assert "the" not in tokens and "and" not in tokens
    assert "quick" in tokens and "fox" in tokens


def test_stemming_collapses_inflections():
    pre = Preprocessor(remove_stopwords=False, stem=True)
    # running / runs should collapse to a common stem
    run_stem = pre.process("running")
    runs_stem = pre.process("runs")
    assert run_stem == runs_stem


def test_empty_text():
    pre = Preprocessor()
    assert pre.process("") == []
    assert pre.process("!!! ??? ...") == []
