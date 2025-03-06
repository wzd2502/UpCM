import numpy as np
from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.Node import Node
import pandas as pd
import os
from math import log
from pgmpy.estimators import BaseEstimator
import random

random.seed(0)

rootpath = os.getcwd()

def csv2generalgraph(networkname: str) -> GeneralGraph:
    g = GeneralGraph([])
    node_map = {}
    files = pd.read_csv(os.path.join(rootpath, 'Networks', networkname + '.csv'), header=0)

    for node in files[:]['Variable 1'].values.tolist():
        if node not in node_map.keys():
            node_map[node] = GraphNode(node)
            g.add_node(node_map[node])
    for node in files[:]['Variable 2'].values.tolist():
        if node not in node_map.keys():
            node_map[node] = GraphNode(node)
            g.add_node(node_map[node])

    for i in range(files.shape[0]):
        fromnode = node_map[files.iloc[i][1]]
        tonode = node_map[files.iloc[i][3]]
        g.add_directed_edge(fromnode, tonode)

    return g

def csv2dataset(networkname: str, num_of_dataset: int):
    # np.random.seed(1123)

    files = pd.read_csv(os.path.join(rootpath, 'Datasets', networkname + '.csv'), header=0)
    dataset = files.sample( n = num_of_dataset, replace = False)
    for i in range(dataset.shape[1]):
        indexs = dataset.iloc[:,i].values.tolist()
        indexs = list(set(indexs))
        dataset.iloc[:,i].replace(indexs, list(range(0,len(indexs))), inplace = True)

    return dataset

