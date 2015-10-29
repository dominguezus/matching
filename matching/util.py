


def shingle(s, k):
    """Generate k-length shingles of string s."""
    k = min(len(s), k)
    for i in range(len(s) - k + 1):
        yield tuple(s[i:i + k])  # Returning tuple allows making set of token lists


def tokens(name):
    ''' Return a generator that gets tokens from the input name.  Should probably
    do this with NLTK eventually '''
    return (t.lower() for t in re.split('\W+', name))

