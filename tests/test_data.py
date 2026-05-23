from ntkmirror.data import Example, batches


def test_batches():
    xs = [Example("p", " c") for _ in range(5)]
    assert [len(b) for b in batches(xs, 2)] == [2, 2, 1]
