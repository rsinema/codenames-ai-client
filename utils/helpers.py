from itertools import chain, combinations

from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer(language="english")


def powerset(iterable, rng=range(2, 4)):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in rng)


def isValid(word, board_words):
    word_stem = stemmer.stem(word)
    for w in board_words:
        curr_stem = stemmer.stem(w)
        if word_stem == curr_stem:
            return False
    return "_" not in word and not word.isupper() and word.isalpha()


def flatten(t):
    return [item for sublist in t for item in sublist]
