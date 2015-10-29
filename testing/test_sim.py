import pytest


from matching import similarity

test_strings = [
    ( 'acme traffic signal company', 'acme traffic signal co', 0.875),
    ( 'acme rocket powered products', 'acme traffic signal co.', 0.45)
]

ngram_test_strings = [
    ( 'acme', 'acme', 1.0),
    ( 'acme t', 'acme u', 2.0/3.0)
]


@pytest.mark.parametrize("s1,s2,expected", test_strings)
def test_jaccard(s1, s2, expected):
    jsim = similarity.jaccard(s1, s2) 
    assert jsim == expected


@pytest.mark.parametrize("s1,s2,expected", ngram_test_strings)
def test_jaccard_kgram(s1, s2, expected):
    jsim = similarity.jaccard_kgram(s1, s2) 
    assert jsim == expected


def test_score_strings():
    scores = similarity.score_strings('acme traffic signal company', [ts[0] for ts in test_strings])
    assert next(scores)[1] == 1.0
    assert next(scores)[1] == 0.5
