import numpy as np
import pandas as pd
from causallearn.utils.TXT2GeneralGraph import txt2generalgraph
from causallearn.search.ScoreBased.GES import ges
from causallearn.graph.SHD import SHD
from causallearn.utils.PDAG2DAG import pdag2dag
import random
import os
from sklearn.metrics import normalized_mutual_info_score
from MOEAD.DataProcess import csv2generalgraph
import csv
# from pyCausalFS.pyBN.learning.structure.hybrid.mmpc import mmpc

rootpath = os.getcwd()

random.seed(0)
class CausalGraphObject:

    def __init__(self, name):
        """

        :param name:
        :param num_data:
        :param score_type:
        """
        # self.benchmark_graph = txt2generalgraph(os.path.join(rootpath, 'Networks', name + '.txt'))
        self.benchmark_graph = csv2generalgraph(networkname=name)
        self.name = name

    def read_constraints(self, cur_name):

        csv_file = os.path.join(rootpath, 'Constraints', self.name,  'Constraints_' + cur_name)
        constraints = pd.read_csv(csv_file, header=0)
        constraints_matrix = np.zeros((constraints.shape[0],4), dtype=int)

        for i in range(constraints.shape[0]):
            constraints_matrix[i, 0] = self.benchmark_graph.node_map[self.benchmark_graph.get_node(constraints.iloc[i, 1])]
            constraints_matrix[i, 1] = self.benchmark_graph.node_map[self.benchmark_graph.get_node(constraints.iloc[i, 2])]
            constraints_matrix[i, 2] = i
            constraints_matrix[i, 3] = int(constraints.iloc[i, 3])

        return  constraints_matrix

    def initial_constraints(self, neg_times, pos_con_rate):
        """
        :param neg_times: 
        :param pos_con_rate: 
        :return: 
        """
        # n: the num of total nodes;
        # p: the num of true constrained edges;
        # q: the num of false constrained edges;
        n = self.benchmark_graph.get_num_nodes()
        p = round(pos_con_rate * self.benchmark_graph.get_num_edges())
        q = neg_times * p

        true_dag_matrix = self.benchmark_graph.graph.copy()
        # i->j = -1 imply the directed edge. true_dag_matrix is a dag!
        true_dag_matrix[np.where(true_dag_matrix == 1)] = 0
        true_dag_matrix[np.where(true_dag_matrix == -1)] = 1
        ###false sample method1:
        # false_dag_matrix = np.ones((n, n)) - np.eye(n) - true_dag_matrix
        false_dag_matrix = np.ones((n, n)) - np.eye(n) - true_dag_matrix
        # false_dag_matrix = np.triu(false_dag_matrix, k = 0)

        true_edge_index = np.where(true_dag_matrix == 1)
        false_edge_index = np.where(false_dag_matrix == 1)
        constraint_matrix = np.empty((p + q, 4), dtype= int )

        # c: the index set of sampled true constrained edges;
        # v: the index set of sampled false constrained edges;
        # np.random.seed(15893)
        c = random.sample(range(len(true_edge_index[1])), p)
        v = random.sample(range(len(false_edge_index[1])), q)

        k = 0
        for i in c:
            constraint_matrix[k, :] = np.array([true_edge_index[0][i], true_edge_index[1][i], k, 1])
            k += 1
        for j in v:
            constraint_matrix[k, :] = np.array([false_edge_index[0][j], false_edge_index[1][j], k, 0])
            k += 1

        return constraint_matrix

    def get_pc_constraint_matrix(self, constraint_matrix):

        # m: num of nodes;
        # n: num of constraints;
        n = constraint_matrix.shape[0]
        m = self.benchmark_graph.get_num_nodes()

        # PC: parent_child_con__matrix
        pc_matrix = np.zeros((m, m, n))

        # follow the expression of weight-matrix in causal-learn package
        for i in range(n):
            pc_matrix[constraint_matrix[i, 0], constraint_matrix[i, 1], i] = -1
            pc_matrix[constraint_matrix[i, 1], constraint_matrix[i, 0], i] = 1

        return pc_matrix

    def GetNMIofConstraints(self, dataset, constraint_matrix):

        nmi = []
        for i in range(constraint_matrix.shape[0]):
            index1 = constraint_matrix[i, 0]
            index2 = constraint_matrix[i, 1]
            nmi.append(normalized_mutual_info_score(dataset.iloc[:, index1],  dataset.iloc[:, index2]))
        # print(constraint_matrix)
        # print(nmi)

    def WriteConstraintsTocsv(self, constraint_matrix, cur_name):

        constraints = pd.DataFrame(columns= ['ID', 'Parent', 'Child', 'TrueOrFalse'])
        for i in range(constraint_matrix.shape[0]):
            index1 = constraint_matrix[i, 0]
            index2 = constraint_matrix[i, 1]
            index3 = constraint_matrix[i, 2]
            if constraint_matrix[i, 3] == 0:
                index4 = 'false'
            else:
                index4 = 'true'

            data = {'ID': index3, 'Parent': self.benchmark_graph.get_node_names()[index1], 'Child': self.benchmark_graph.get_node_names()[index2], 'TrueOrFalse': index4}
            constraints = pd.concat([constraints, pd.DataFrame([data])], ignore_index=True)
        constraints.to_csv(path_or_buf=os.path.join(rootpath, 'Constraints', self.name, 'Constraints_' + cur_name + '.csv'), index=False)

        return constraints




