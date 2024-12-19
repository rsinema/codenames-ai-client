from .constants import Team


class SpymasterCombo:
    def __init__(self):
        self.scores = []
        self.max_clue = None
        self.max_sim = -9999

    def addOption(self, clue, sim):
        self.scores.append(sim)
        if self.max_sim < sim:
            self.max_sim = sim
            self.max_clue = clue

    def getAvgSim(self):
        return sum(self.scores) / len(self.scores)


class BaseSpymaster:
    def __init__(self, assoc):
        self.assoc = assoc

    def makeClue(self, board, team: Team):
        raise NotImplementedError
