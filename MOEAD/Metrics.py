from causallearn.graph import GeneralGraph


class EdgeConfusion:
    """
    Compute the adjacency confusion between two graphs.
    """
    __adjFn = 0
    __adjTp = 0
    __adjFp = 0
    __adjTn = 0

    def __init__(self, truth: GeneralGraph, est: GeneralGraph):
        """
        Compute and store the edge confusion between two dags.

        Parameters
        ----------
        truth :
            Truth GeneralGraph : DAG
        est :
            Estimated GeneralGraph : DAG
        """
        nodes = truth.get_nodes()
        for i in range(len(nodes)):
            for j in range(len(nodes)):
                if i != j:
                    if est.graph[i, j] == -1 and est.graph[j, i] == 1:
                        estAdj = True
                    else:
                        estAdj = False
                    if truth.graph[i, j] == -1 and truth.graph[j, i] == 1:
                        truthAdj = True
                    else:
                        truthAdj = False

                    if truthAdj and not estAdj:
                        self.__adjFn = self.__adjFn + 1
                    elif estAdj and not truthAdj:
                        self.__adjFp = self.__adjFp + 1
                    elif estAdj and truthAdj:
                        self.__adjTp = self.__adjTp + 1
                    elif not estAdj and not truthAdj:
                        self.__adjTn = self.__adjTn + 1

    def get_adj_tp(self):
        return self.__adjTp

    def get_adj_fp(self):
        return self.__adjFp

    def get_adj_fn(self):
        return self.__adjFn

    def get_adj_tn(self):
        return self.__adjTn

    def get_adj_precision(self):
        if self.__adjTp + self.__adjFp != 0:
            return self.__adjTp / (self.__adjTp + self.__adjFp)
        else:
            return 0

    def get_adj_recall(self):
        if self.__adjTp + self.__adjFp != 0:
            return self.__adjTp / (self.__adjTp + self.__adjFn)
        else:
            return 0
