from causallearn.search.ScoreBased.GES import ges
from causallearn.search.ConstraintBased.PC import pc
from causallearn.score.LocalScoreFunction import local_score_BIC
# from pyCausalFS.pyBN.learning.structure.hybrid.mmpc import mmpc
import pandas as pd
import numpy as np
from pgmpy.estimators.HillClimbSearch import HillClimbSearch, BIC
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from pgmpy.estimators.MmhcEstimator import MmhcEstimator
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.Graph import Graph
from copy import deepcopy
#hc
import random
import os
rootpath = os.getcwd()

def GetLearnedGraph(network, method, cur_name, hasconstraints, example_CL: GeneralGraph):

    g = GeneralGraph([])
    node_map = {}
    if hasconstraints:
        files = pd.read_csv(os.path.join(rootpath, 'Networks', network, method, f'{method}con_' + cur_name), header=0)
    else:
        files = pd.read_csv(os.path.join(rootpath, 'Networks', network, method, f'{method}_' + cur_name), header=0)

    for node in example_CL.get_node_names():
        if node not in node_map.keys():
            node_map[node] = GraphNode(node)
            g.add_node(node_map[node])


    for i in range(files.shape[0]):
        fromnode = node_map[files.iloc[i][1]]
        tonode = node_map[files.iloc[i][3]]
        g.add_directed_edge(fromnode, tonode)

    return g

def GetGesGraph(networkname, repeat, hasconstraints, example_CL:GeneralGraph):

    g = GeneralGraph([])
    node_map = {}
    if hasconstraints:
        sign = 'withconstraints'
    else:
        sign = 'withoutconstraints'
    files = pd.read_csv(os.path.join(rootpath, 'Networks', 'GES', networkname, sign, f'{networkname}_GES_{repeat}.csv'), header=0)

    for node in example_CL.get_node_names():
        if node not in node_map.keys():
            node_map[node] = GraphNode(node)
            g.add_node(node_map[node])


    for i in range(files.shape[0]):
        fromnode = node_map[files.iloc[i][1]]
        tonode = node_map[files.iloc[i][3]]
        g.add_directed_edge(fromnode, tonode)

    return g

def GetPCGraph(dataset, example_CL:GeneralGraph):

    nodes = example_CL.get_nodes()
    pdag = GeneralGraph(nodes)
    dataset = dataset[example_CL.get_node_names()]
    dataset = np.array(dataset)
    pdag_matrix = pc(dataset, indep_test='gsq').G
    pdag.graph = pdag_matrix.graph
    pdag.edges = pdag.get_graph_edges()

    # it is a cpdag
    return pdag

def GetHCGraph(dataset, example_CL:GeneralGraph):

    nodes = example_CL.get_nodes()
    pdag = GeneralGraph(nodes)

    est = HillClimbSearch(dataset)
    best_model = est.estimate(scoring_method=BIC(dataset))
    uds = best_model.edges()
    for ud in uds:
        fromnode = pdag.get_node(ud[0])
        tonode = pdag.get_node(ud[1])
        pdag.add_directed_edge(fromnode, tonode)

    # est = MmhcEstimator(data)
    # best_model = est.estimate()
    # uds = best_model.edges()
    # for ud in uds:
    #     pdag.add_bidirected_edge(nodes[ud[0]], nodes[ud[1]])

    return pdag

def GetEmptyGraph(dataset,example_CL:GeneralGraph):

    nodes = example_CL.get_nodes()
    pdag = GeneralGraph(nodes)
    return pdag

def GetPCGraph_Con(dataset, example_CL:GeneralGraph, constraint_matrix: np.array([[]])):

    dataset = dataset[example_CL.get_node_names()]
    BGKL = BackgroundKnowledge()
    nodes = example_CL.get_nodes()
    pdag = GeneralGraph(nodes)
    for i in range(constraint_matrix.shape[0]):
        from_index = constraint_matrix[i, 0]
        to_Index = constraint_matrix[i, 1]
        BGKL.add_required_by_node(nodes[from_index], nodes[to_Index])
    current_graph = pc(np.array(dataset), background_knowledge=BGKL, indep_test='chisq').G
    pdag.graph = current_graph.graph
    pdag.edges = pdag.get_graph_edges()

    return pdag

def GetMahcGraph(networkname, example_CL:GeneralGraph):

    g = GeneralGraph([])
    node_map = {}
    files = pd.read_csv(os.path.join(rootpath, 'Networks\\MAHC', networkname + '.csv'), header=0)

    for node in example_CL.get_node_names():
        if node not in node_map.keys():
            node_map[node] = GraphNode(node)
            g.add_node(node_map[node])
    # for node in files[:]['Variable 2'].values.tolist():
    #     if node not in node_map.keys():
    #         node_map[node] = GraphNode(node)
    #         g.add_node(node_map[node])
    # g.set_nodes(example_CL.get_nodes())

    for i in range(files.shape[0]):
        fromnode = node_map[files.iloc[i][1]]
        tonode = node_map[files.iloc[i][3]]
        g.add_directed_edge(fromnode, tonode)

    return g