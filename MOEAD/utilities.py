from copy import deepcopy

import numpy as np

from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.utils.DAG2CPDAG import dag2cpdag
from causallearn.utils.PDAG2DAG import pdag2dag
from causallearn.graph.Dag import Dag
from causallearn.score.LocalScoreFunction import local_score_BIC, local_score_BDeu
from pgmpy.estimators.HillClimbSearch import BIC,BDeu
from pgmpy.models import BayesianNetwork
from pgmpy.metrics.metrics import correlation_score
from pgmpy.metrics.metrics import log_likelihood_score
from scipy.linalg import expm

import logging

# 设置 pgmpy 的日志级别为 WARNING
logging.getLogger('pgmpy').setLevel(logging.WARNING)


def PDAG2DAG_NonC(G: GeneralGraph, Cm):
    """

    :param G: current pdag (graph type)
    :param Cm: constraint_matrix, is not a general graph!
    :return: nonconsistent extension
    """

    nodes = G.get_nodes()

    std4 = deepcopy(G.graph)
    std5 = ((np.abs(std4) - std4)) / 2
    std6 = expm(std5)

    # first create a DAG that contains all the directed edges in PDAG
    Gd = deepcopy(G)
    edges = Gd.get_graph_edges()
    for edge in edges:
        if not ((edge.endpoint1 == Endpoint.ARROW and edge.endpoint2 == Endpoint.TAIL) or (
                edge.endpoint1 == Endpoint.TAIL and edge.endpoint2 == Endpoint.ARROW)):
            Gd.remove_edge(edge)

    Gp = deepcopy(G)
    inde = np.zeros(Gp.num_vars, dtype=np.dtype(int))  # index whether the ith node has been removed. 1:removed; 0: not
    mini_open = False
    old_inde = inde.copy()
    break_sign = 0
    while 0 in inde:
        if break_sign > 2:
            print('Have cycle')
            # raise TypeError("Have cycle!`")
            return Gd, True
        for i in range(Gp.num_vars):
            if inde[i] == 0:
                sign = 0
                zd1 = np.intersect1d(np.where(Gp.graph[:, i] == 1)[0], np.where(inde == 0)[0])
                if len(zd1) == 0:  # Xi has no out-going edges or the out-going edges come from Constraints
                    sign = sign + 1
                    Nx = np.intersect1d(
                        np.intersect1d(np.where(Gp.graph[:, i] == -1)[0], np.where(Gp.graph[i, :] == -1)[0]),
                        np.where(inde == 0)[0])  # find the neighbors of Xi in P
                    Ax = np.intersect1d(np.union1d(np.where(Gp.graph[i, :] == 1)[0], np.where(Gp.graph[:, i] == 1)[0]),
                                        np.where(inde == 0)[0])  # find the adjacent of Xi in P
                    Ax = np.union1d(Ax, Nx)
                    if len(Nx) > 0:
                        if check2(Gp, Nx, Ax,mini_open):  # only add new v-structure when index->i is covered, now, it is not necessary
                            sign = sign + 1
                    else:
                        sign = sign + 1
                if sign == 2:
                    # for each undirected edge Y-X in PDAG, insert a directed edge Y->X in G
                    for index in np.intersect1d(np.where(Gp.graph[:, i] == -1)[0], np.where(Gp.graph[i, :] == -1)[0]):
                        # if not Gd.is_ancestor_of(nodes[i], nodes[index]): # do not create a cyclic. use Gd instead of Gp.
                        #     Gd.add_edge(Edge(nodes[index], nodes[i], Endpoint.TAIL, Endpoint.ARROW))
                        # else:
                        #     Gd.add_edge(Edge(nodes[i], nodes[index], Endpoint.TAIL, Endpoint.ARROW))
                        # Gd.reconstitute_dpath(Gd.get_graph_edges()) ## need update the dpath-info
                        # if Gd.dpath[i, index] == 1 and Gd.dpath[index, i] == 1:
                        #     return G, True
                        Gd.add_edge(Edge(nodes[index], nodes[i], Endpoint.TAIL, Endpoint.ARROW))
                    inde[i] = 1
        if np.array_equal(old_inde, inde):
            mini_open = True
            break_sign += 1
            ### use CPDAG
            # return Gd, True
        else:
            old_inde = inde.copy()
            mini_open = False

    return Gd, False

def check2(G: GeneralGraph, Nx, Ax, mini_open):
    # s = 1
    # for i in range(len(Nx)):
    #     j = np.delete(Ax, np.where(Ax == Nx[i])[0])
    #     # test if contain the directed path Nx[i] --> j(if do not add new v-structure will lead to cycle, 'corvered' by a path), G.dpath is not convinced, we should use expotienal matrix
    #     zd3 = len(np.where(std6[Nx[i], j] != 0)[0])
    #     # test if contain x1-->x2---x3<--x4 structure
    #     zd4 = len(np.where(G.graph[nowi, :] == 1)[0])
    #     zd5 = len(np.where(G.graph[Nx[i], :] == 1)[0])
    #     # if zd3 != 0 or (zd4 != 0 and zd5 != 0):
    #     if zd4 != 0 and zd5 != 0:
    #         break
    #     else:
    #         zd6 = len(np.where(G.graph[j, Nx[i]] == 0)[0])
    #         if zd6 != 0:
    #              s = 0
    #         break
    # return s

    s = 1
    for i in range(len(Nx)):
        j = np.delete(Ax, np.where(Ax == Nx[i])[0])
        if len(np.where(G.graph[Nx[i], j] == 0)[0]) != 0:
            if not mini_open:
                s = 0
                break
    return s


def Matrix2Graph(Gm: np.array([[]]), oldgraph:GeneralGraph) -> GeneralGraph:
    '''

    :param Gm: The new matrix of graph
    :param oldgraph: the old graph
    :return: updated graph
    '''
    oldgraph.graph = deepcopy(Gm)
    oldgraph.edges = oldgraph.get_graph_edges()
    oldgraph.reconstitute_dpath(oldgraph.get_graph_edges())
    return oldgraph

def WeightMatrix_Con(wm: np.array([[]]), cm: np.array([[]]), decision_vector: np.array([])):
    '''

    :param wm: the matrix of graph (m * m)
    :param cm: constraint matrix (n * 4 )
    :param decision_vector: which constraint should be contained in wm (n * 1)
    :return: updated matrix (m * m)
    '''
    for i in range(len(decision_vector)):
        if decision_vector[i] != 0:
        # correct and wrong decision vector should all be deleted
            wm[cm[i, 0], cm[i, 1]] = 0
            wm[cm[i, 1], cm[i, 0]] = 0
    return wm

def InitialG2FinalG(decision_vector, initial_graph: GeneralGraph, constraint_matrix, pc_matrix) -> GeneralGraph:

    '''

    :param decision_vector:
    :param initial_graph:
    :param constraint_matrix: n * 4
    :param pc_matrix: m * m * n
    :return: the updated graph based on decision_vector and initial graph
    '''
    print(decision_vector)
    nodes = initial_graph.get_nodes()
    current_graph = deepcopy(initial_graph)
    for j in range(len(decision_vector)):
        if decision_vector[j] != 0:
            a = 1
            fromnode = int(constraint_matrix[j, 0])
            tonode = int(constraint_matrix[j, 1])
            current_graph.graph[fromnode, tonode] = -1
            current_graph.graph[tonode, fromnode] = 1
        else:
            fromnode = int(constraint_matrix[j, 0])
            tonode = int(constraint_matrix[j, 1])
            if current_graph.graph[tonode, fromnode] == -1:
                current_graph.graph[fromnode, tonode] = 1
            else:
                current_graph.graph[fromnode, tonode] = 0
                current_graph.graph[tonode, fromnode] = 0
    current_graph.edges = current_graph.get_graph_edges()
    current_graph.reconstitute_dpath(current_graph.get_graph_edges())
    # current_graph.reconstitute_dpath(current_graph.edges)
    # uuu = deepcopy(current_graph)
    # if current_graph.exists_directed_cycle():
    #     raise TypeError("Have Cyclic!`")
    # print(current_graph)
    current_graph1, illegal = PDAG2DAG_NonC(current_graph, np.sum(pc_matrix, axis=2))
    # current_graph2 = pdag2dag(current_graph)
    # if illegal:
    #     # return current_graph, False
    #     raise TypeError("Have illegal!`")
    # print(current_graph)
    # current_graph_dag = Dag(current_graph.get_nodes())
    # for edge in current_graph.edges:
    #     fromnode = edge.node1
    #     tonode = edge.node2
    #     current_graph_dag.add_directed_edge(fromnode, tonode)

    ### important!
    # current_graph = dag2cpdag(current_graph)
    # print(current_graph)
    return current_graph1, not illegal

def GetBicScoreOfGraph(RequiredGraph:GeneralGraph, dataset, sign:bool) ->float:

    # if input is cpdag, find its consistent extension
    dag = deepcopy(RequiredGraph)
    if sign:
        dag = pdag2dag(dag)

    parameters = {}
    parameters['lambda_value'] = 2
    score = 0
    # mix_dag = dag.graph
    # for i in range(mix_dag.shape[1]):
    #     pai = np.where(mix_dag[i, :] >= 1)
    #     # score += local_score_BIC(dataset, i, pai[0], parameters=parameters)
    #     score += bic_score(dataset, i, pai[0], parameters = parameters, states_count= states_count)
    #     # score += local_score_BDeu(dataset, i, pai[0])
    # score = float(score)

    edge_list = []
    node_list = []
    BN = BayesianNetwork()
    for edge in dag.get_graph_edges():
        fromnode = edge.node1.get_name()
        tonode = edge.node2.get_name()
        edge_list.append((fromnode, tonode))
    for node in dag.get_node_names():
        node_list.append(node)
    # print(edge_list)
    BN.add_nodes_from(node_list)
    BN.add_edges_from(edge_list)
    score = BDeu(dataset, equivalent_sample_size= dataset.shape[0]).score(BN)
    # score = BIC(dataset).score(BN)

    return score


def GetCorScoreOfGraph(RequiredGraph:GeneralGraph, dataset) ->float:

    dag = deepcopy(RequiredGraph)
    edge_list = []
    node_list = []
    BN = BayesianNetwork()
    for edge in dag.get_graph_edges():
        fromnode = edge.node1.get_name()
        tonode = edge.node2.get_name()
        edge_list.append((fromnode, tonode))
    for node in dag.get_node_names():
        node_list.append(node)
    # print(edge_list)
    BN.add_nodes_from(node_list)
    BN.add_edges_from(edge_list)
    score = correlation_score(BN, dataset, test="chi_square", significance_level=0.05)
    return score

def GetLogLikeScoreOfGraph(RequiredGraph:GeneralGraph, dataset) ->float:

    dag = deepcopy(RequiredGraph)
    edge_list = []
    node_list = []
    BN = BayesianNetwork()
    for edge in dag.get_graph_edges():
        fromnode = edge.node1.get_name()
        tonode = edge.node2.get_name()
        edge_list.append((fromnode, tonode))
    for node in dag.get_node_names():
        node_list.append(node)
    # print(edge_list)
    BN.add_nodes_from(node_list)
    BN.add_edges_from(edge_list)
    # score = correlation_score(BN, dataset, test="chi_square", significance_level=0.05)
    score = log_likelihood_score(BN, dataset)
    return score

def preorientate(qq:GeneralGraph, constraint_matrix:np.array([[]])) -> GeneralGraph:

    gg = deepcopy(qq)
    for i in range(constraint_matrix.shape[0]):
        fromnode = int(constraint_matrix[i][0])
        tonode = int(constraint_matrix[i][1])
        if gg.graph[fromnode, tonode] == -1 and gg.graph[tonode, fromnode] == -1:
            gg.graph[tonode, fromnode] = 1

    gg.edges = gg.get_graph_edges()
    return gg

def acyclic(dag:np.array([[]])):

    std4 = deepcopy(dag)
    std5 = ((np.abs(std4) + std4).T) / 2
    std6 = np.trace(expm(std5)) - dag.shape[0]

    if std6 != 0:
        return False
    else:
        return True