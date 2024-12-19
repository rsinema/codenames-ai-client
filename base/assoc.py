class Assoc:
    def __init__(self):
        pass

    def getAssocs(self, pos, neg, topn):
        raise NotImplementedError

    def preprocess(self, w):
        raise NotImplementedError

    def clearCache(self):
        return
