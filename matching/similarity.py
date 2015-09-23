
import math
import functools
import itertools
import re
import jellyfish
from collections import Counter, defaultdict

'''
@todo - MinHash/LSH for larger data sets
      - Look at actual n-gram frequencies and do something like MI or K-L
'''


def jaccard(s1, s2):
    ''' Jaccard similarity score.  (1 - jaccard) is a valid distance metric '''
    a = set(s1)
    b = set(s2)
    if len(a) == 0 and len(b) == 0:
        return 0
    return len(a & b) / len(a | b)


def jaccard_kgram(s1, s2, k=2):
    ''' Converts inputs to k-gram/shingles before computing Jaccard similarity '''
    return jaccard(frozenset(shingle(s1, k)), frozenset(shingle(s2, k)))


def shingle(s, k):
    """Generate k-length shingles of string s."""
    k = min(len(s), k)
    for i in range(len(s) - k + 1):
        yield tuple(s[i:i + k])  # Returning tuple allows making set of token lists

def score_string(name, other_name, scoring_func=jaccard, analyzer_func=lambda x: x):
    return scoring_func(analyzer_func(name), analyzer_func(other_name))


def tokens(name):
    ''' Return a generator that gets tokens from the input name.  Should probably
    do this with NLTK eventually '''
    return (t.lower() for t in re.split('\W+', name))


def rankings(t1, t2, sim_func=jaccard):
    ''' Given two sets of tokens, rank each token in set 1 vs the tokens in set 2 by similarity '''
    jws = [(p, sim_func(p[0], p[1])) for p in itertools.product(t1, t2)]
    return ({n: sorted([(t[1], s) for t, s in jws if t[0] == n], key=lambda x: x[1], reverse=True) for n in t1},
            {n: sorted([(t[0], s) for t, s in jws if t[1] == n], key=lambda x: x[1], reverse=True) for n in t2})


def match_tokens(t1_rankings, t2_rankings):
    ''' Find the optimal pairing of tokens.  This is basically a version of the Gayle-Shapley
    algorithm '''
    xmatched = {}
    r1 = {n: [t[0] for t in r] for n, r in t1_rankings.items()}
    r2 = {n: [t[0] for t in r] for n, r in t2_rankings.items()}
    unmatched_tokens = list(r1.keys())
    while unmatched_tokens:
        token = unmatched_tokens.pop(0)
        match_token = r1[token].pop(0)
        if match_token not in xmatched:
            xmatched[match_token] = token
        else:
            curr_match = xmatched[match_token]
            if r2[match_token].index(token) < r2[match_token].index(curr_match):
                xmatched[match_token] = token
                if r1[curr_match]:  # pragma: no cover
                    unmatched_tokens.append(curr_match)
            else:
                if r1[token]:
                    unmatched_tokens.append(token)

    return {v: k for k, v in xmatched.items()}


def token_match(tokens1, tokens2, sim_func=jaccard):
    ''' Rank and pair tokens '''
    r1, r2 = rankings(tokens1, tokens2, sim_func)
    return match_tokens(r1, r2)


def term_counts(names):
    ''' Build a counter of tokens in the set of all possible names '''
    return Counter(itertools.chain(*(tokens(n) for n in names)))


def soft_idf(name1, name2, counts, sim_func=jaccard):
    ''' Implementation of Soft IDf matching.  Finds the best pairing of tokens in the two
    names based on some similarity metric, and then scores that pairing based on the token
    similarity and the inverse document frequency (IDF) of the terms. '''
    name1_tokens = list(tokens(name1))
    name2_tokens = list(tokens(name2))
    tok_match = token_match(name1_tokens, name2_tokens, sim_func)

    idf = {t: math.log(len(counts) / (counts[t] + 1)) for t in name1_tokens}
    idf.update({t: math.log(len(counts) / (counts[t] + 1)) for t in name2_tokens})

    def sx(tk, idf):
        return math.sqrt(sum([(idf[t] / len(tk)) ** 2 for t in tk if t]))

    return sum(sim_func(a, b) * idf[a] / len(name1_tokens) * idf[b] / len(name2_tokens)
               for a, b in tok_match.items()) / (sx(name1_tokens, idf) * sx(name2_tokens, idf))


def gauss_distance(t, u, s):
    return math.exp(-(t - u) ** 2 / (2 * s ** 2))


def build_entity_name_matcher(names_list, key=None):
    ''' Returns a function that can be used to match entity names based on the SoftIDF metric '''
    names = [key(x) for x in names_list] if key else names_list
    f = functools.partial(soft_idf, counts=term_counts(names), sim_func=jellyfish.jaro_winkler)

    def match_by_name(advisor, advisor_to_match, key=key):
        name1 = key(advisor) if key else advisor
        name2 = key(advisor_to_match) if key else advisor_to_match
        return score_string(name1, name2, scoring_func=f, analyzer_func=lambda x: x.lower())

    return match_by_name

