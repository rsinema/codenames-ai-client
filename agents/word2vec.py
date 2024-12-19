from collections import defaultdict
from typing import Optional

from gensim.models import KeyedVectors

from base.assoc import Assoc
from base.constants import Team
from base.spymaster import BaseSpymaster, SpymasterCombo
from utils.helpers import isValid, powerset


class ModelSingleton:
    """Word2Vec model singleton."""

    _instance: Optional[KeyedVectors] = None

    @classmethod
    def get_model(cls) -> KeyedVectors:
        if not cls._instance:
            print("loading w2v model")
            cls._instance = KeyedVectors.load_word2vec_format(
                "GoogleNews-vectors-negative300.bin", binary=True, limit=500000
            )
        return cls._instance


class W2VAssoc(Assoc):
    def __init__(self, debug=False):
        super().__init__()
        self.model = ModelSingleton.get_model()
        self.debug = debug

    def getAssocs(self, pos, neg, topn):
        if self.debug:
            print("W2V", pos, neg)
        return self.model.most_similar(
            positive=pos, negative=neg, topn=topn, restrict_vocab=50000
        )

    def preprocess(self, w):
        try:
            self.model.key_to_index[w]
        except KeyError:
            w = "_".join([part[0].upper() + part[1:] for part in w.split("_")])
        return w


class W2VSpymaster(BaseSpymaster):
    def __init__(self, assoc, debug=False):
        super().__init__(assoc)
        self.debug = debug

    def makeClue(self, board, team: Team):
        board_words = set(
            [item for sublist in list(board.values()) for item in sublist]
        )

        # Get my team's words and opponent's words
        my_words = board["U" if team == Team.BLUE else "R"]
        opponent_words = board["R" if team == Team.BLUE else "U"]

        # Combine opponent's words with neutral and assassin words
        neg = board["N"] + board["A"] + opponent_words
        pos = my_words

        neg = [self.assoc.preprocess(w) for w in neg]
        pos = [self.assoc.preprocess(w) for w in pos]

        self.assoc.clearCache()

        combos = defaultdict(SpymasterCombo)

        if len(pos) == 1:
            pow_set = [tuple(pos)]
        else:
            pow_set = powerset(pos)
        for combo in pow_set:
            curr = self.assoc.getAssocs(list(combo), neg, 10)

            any_added = False
            for clue, sim in curr:
                clue = clue.lower()
                if isValid(clue, board_words):
                    combos[combo].addOption(clue, sim)
                    any_added = True
            if not any_added:
                print("NONE ADDED:", combo, [clue for clue, sim in curr])

        if self.debug:
            print("mH combos:", combos)

        max_avg_sim = -9999
        max_combo = None

        if not combos.keys():
            print("NO CLUE!", board, team, pos, neg)
            return ("None", 1)

        for combo in combos.keys():
            avg_sim = combos[combo].getAvgSim()
            if self.debug:
                print("mH combo+avg\t:", combo, avg_sim)
            if max_avg_sim < avg_sim:
                max_avg_sim = avg_sim
                max_combo = combo

        if self.debug:
            print("mH max_combo:", max_combo)
        return (combos[max_combo].max_clue, len(max_combo)), max_combo
