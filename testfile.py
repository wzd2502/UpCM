import numpy as np
import pandas as pd
from causallearn.search.ScoreBased.GES import ges
from causallearn.search.ScoreBased.ExactSearch import bic_exact_search
from causallearn.utils.TXT2GeneralGraph import txt2generalgraph
import os
import random
from causallearn.utils.PDAG2DAG import pdag2dag
from causallearn.utils.DAG2CPDAG import dag2cpdag
from causallearn.score.LocalScoreFunction import local_score_BDeu, local_score_BIC
from minepy import MINE
from sklearn.metrics import normalized_mutual_info_score
from causallearn.graph.Dag import Dag
from causallearn.graph.Node import Node
from scipy.linalg import expm
from MOEAD.utilities import PDAG2DAG_NonC
from MOEAD.utilities import *
# from MOEAD.DataProcess import *
import networkx as nx
from pgmpy.estimators.StructureScore import BicScore
from pgmpy.models import BayesianNetwork
from MOEAD.CompareAndPlot import *
import os

# ff = csv2generalgraph('ASIA')
# print(ff)
#
rootpath = os.getcwd()
# filepath = os.path.join(rootpath, 'BayesMethodsResults', 'ALARM_100.csv')
# constraintpath = os.path.join(rootpath, 'Constraints', 'ALARM', '100')
# CompareBayesMethods(filepath, constraintpath,'ALARM', '100', 0.2, 1)
# cc = txt2generalgraph(os.path.join(rootpath, 'Networks', 'graph2.txt'))
# ddg = txt2generalgraph(os.path.join(rootpath, 'Networks', 'constraints'))
# data = np.loadtxt(os.path.join(rootpath,'Datasets','testscore'),skiprows=1)
# data = pd.DataFrame(data[:,[1,2,3]], columns=['X1','X2', 'X3', 'X4'])
# gg,ff = PDAG2DAG_NonC(cc, ddg.graph)
# ee = pdag2dag(cc)
# print(ee)
# print(gg)
# print()

# results = pd.read_csv(os.path.join(rootpath,'Results', 'results.csv'))
# PlotScatter(results)

# dd = pdag2dag(cc)
# print(dd)
# cc.graph[1,2] = 1
# cc.graph[2,1] = -1
# cc.graph[2,3] = 1
# cc.graph[3,2] = -1
# cc.graph[3,1] = 1
# cc.graph[1,3] = -1
# # cc.get_graph_edges()
# cc.reconstitute_dpath(cc.get_graph_edges())
# print(cc.dpath)
# print(cc)
# ee = pdag2dag(cc)
# print(ee)
# ff = dag2cpdag(ee)
# print(ff)

# dataset = csv2dataset('ASIA', 100)
# dd = ges(dataset, 'local_score_BDeu')
# print(dd['G'])
# dd = dd['G']

# data = np.loadtxt(os.path.join(rootpath,'Datasets','testscore'),skiprows=1)
# data = pd.DataFrame(data[:,[1,2,3]], columns=['X1','X2', 'X3'])
# BN = BayesianNetwork()
# BN.add_nodes_from(['X1', 'X2', 'X3'])
# print(BicScore(data).score(BN))
# BN.add_edge('X1', 'X2')
# print(BicScore(data).score(BN))



# generate weigts file
matrix = np.random.rand(100, 2)

# 将第一列限制在0到0.5之间
matrix[:, 1] *= 0.5

# 第二列为1减去第一列的值
matrix[:, 0] = 1 - matrix[:, 1]

file_path = 'weights3.ws'

np.savetxt(os.path.join(rootpath, 'Weights', file_path), matrix)




