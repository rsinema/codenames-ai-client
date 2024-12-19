from itertools import chain, combinations
from typing import Iterable, List, Set

from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer(language="english")


def powerset(iterable: Iterable, rng=range(2, 4)) -> chain:
    """Generate powerset of input iterable within given range."""
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in rng)


def is_valid_clue(word: str, board_words: Set[str]) -> bool:
    """Validate if word can be used as a clue."""
    word_stem = stemmer.stem(word)
    return (
        not any(word_stem == stemmer.stem(w) for w in board_words)
        and "_" not in word
        and not word.isupper()
        and word.isalpha()
    )


def flatten(nested_list: List[List]) -> List:
    """Flatten a list of lists into a single list."""
    return [item for sublist in nested_list for item in sublist]
