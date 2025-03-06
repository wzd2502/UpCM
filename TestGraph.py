# dag_learning.py
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ScoreBased.GES import ges
import numpy as np
from causallearn.utils.TXT2GeneralGraph import txt2generalgraph
from causallearn.graph.GeneralGraph import GeneralGraph
from MOEAD.CompareAndPlot import *
from causallearn.graph.SHD import SHD
import os

rootpath = os.getcwd()
import numpy as np
# learn_dag.py

import numpy as np

def compute_correlations(X):
    """
    Compute pairwise correlations between variables.

    Parameters:
    X (numpy.ndarray): Dataset of shape (m, n).

    Returns:
    numpy.ndarray: Correlation matrix of shape (n, n).
    """
    return np.corrcoef(X, rowvar=False)

def is_conditionally_independent(X, i, j, k_set):
    """
    Check if variables X_i and X_j are conditionally independent given a set of variables.

    Parameters:
    X (numpy.ndarray): Dataset of shape (m, n).
    i (int): Index of the first variable.
    j (int): Index of the second variable.
    k_set (list): Indices of the conditioning set.

    Returns:
    bool: True if conditionally independent, False otherwise.
    """
    # Placeholder for a proper CI test. A more sophisticated CI test is needed in practice.
    return np.abs(np.corrcoef(X[:, i], X[:, j])[0, 1]) < 0.05

def construct_initial_graph(X):
    """
    Construct an initial graph based on CI tests.

    Parameters:
    X (numpy.ndarray): Dataset of shape (m, n).

    Returns:
    numpy.ndarray: Adjacency matrix of the initial graph.
    """
    n = X.shape[1]
    W = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            if i != j and not is_conditionally_independent(X, i, j, []):
                W[i, j] = 1
    return W

def learn_dag(X):
    """
    Learn the Directed Acyclic Graph (DAG) from the dataset.

    Parameters:
    X (numpy.ndarray): Dataset of shape (m, n).

    Returns:
    numpy.ndarray: Adjacency matrix of the learned DAG.
    """
    def has_cycle(v, visited, stack):
        """
        Check if there is a cycle in the graph using DFS.

        Parameters:
        v (int): Current node.
        visited (list): List to keep track of visited nodes.
        stack (list): Stack to keep track of nodes in the current path.

        Returns:
        bool: True if a cycle is detected, False otherwise.
        """
        visited[v] = True
        stack[v] = True
        for u in range(len(W)):
            if W[v, u] == 1:
                if not visited[u]:
                    if has_cycle(u, visited, stack):
                        return True
                elif stack[u]:
                    return True
        stack[v] = False
        return False

    def remove_edge(v, u):
        """
        Remove a directed edge from v to u.

        Parameters:
        v (int): Start node.
        u (int): End node.
        """
        W[v, u] = 0

    def dfs():
        """
        Perform DFS to detect and remove cycles from the graph.
        """
        visited = [False] * len(W)
        stack = [False] * len(W)
        for v in range(len(W)):
            if not visited[v]:
                if has_cycle(v, visited, stack):
                    for u in range(len(W)):
                        if W[v, u] == 1:
                            remove_edge(v, u)
                            if not has_cycle(v, visited, stack):
                                break

    W = construct_initial_graph(X)
    dfs()
    return W



cc = txt2generalgraph(os.path.join(rootpath, 'Networks', 'graph.10.txt'))
data = np.loadtxt(os.path.join(rootpath,'Datasets','graph.10_10000.txt'),skiprows=1)
llmmatrix = learn_dag(data)
llmmatrix += -1 * llmmatrix.T

pcgraph = pc(data).G
# gesgraph = ges(data)

llmgraph = GeneralGraph(cc.get_nodes())
llmgraph.graph = -1 * llmmatrix
llmgraph.edges = llmgraph.get_graph_edges()
llmgraph.reconstitute_dpath(llmgraph.get_graph_edges())

shdllm = SHD(cc, llmgraph).get_shd()
shdpc = SHD(cc, pcgraph).get_shd()
subplotnx(llmgraph, 'llm+AAD')
subplotnx(pcgraph, 'pc')
# gesgraph = SHD(cc, gesgraph).get_shd()
print(shdllm, shdpc)
