from base.assoc import Assoc
from base.constants import Team
from base.spymaster import BaseSpymaster


class MyAssoc(Assoc):
    def __init__(self):
        super().__init__()
        # Initialize your model/embedding here

    def getAssocs(self, pos, neg, topn):
        # Implement your word association logic
        pass

    def preprocess(self, w):
        # Implement any word preprocessing
        pass


class MySpymaster(BaseSpymaster):
    def __init__(self, assoc):
        super().__init__(assoc)

    def makeClue(self, board, team: Team):
        """
        Generate a clue for your team.

        Args:
            board: Dictionary with keys 'R' (red), 'U' (blue), 'N' (neutral), 'A'
                (assassin)
            team: Team.RED or Team.BLUE indicating your team

        Returns:
            tuple: ((clue_word, number_of_words), debug_info)
        """
        pass
