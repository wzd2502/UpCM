import numpy as np
from copy import deepcopy
from moead_framework.problem.problem import Problem
from causallearn.score.LocalScoreFunction import local_score_BIC
from scipy.linalg import expm
from MOEAD.utilities import PDAG2DAG_NonC
from sklearn.metrics import normalized_mutual_info_score, mutual_info_score
from causallearn.graph import GeneralGraph
from MOEAD.utilities import *
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.DAG2CPDAG import dag2cpdag
from scipy.linalg import expm
from MOEAD.CompareAndPlot import subplotnx
import time

class CD_MOEAD_Problem(Problem):
    def __init__(self, num_of_objective, constraint_matrix, InitialGraph: GeneralGraph, data, pc_matrix):

        super().__init__(num_of_objective)
        self.name = 'CausalDiscovery_MOEAD'
        self.constraint_matrix = constraint_matrix
        count = 0
        self.initial_graph = InitialGraph
        self.NumOfGraphs = count
        self.data = data.astype('category')
        self.n = pc_matrix.shape[2] # the len of solution vector
        self.m = pc_matrix.shape[0] # the len of variables in Causal Discovery
        self.time_NonC = 0
        self.repeats_NonC = 0
        self.pc_matrix = pc_matrix

    def f(self, function_id: int, decision_vector: np.ndarray):

        if not isinstance(function_id, int):
            raise TypeError("The expected type of `function_id` is `int`")

        if not isinstance(decision_vector, np.ndarray):
            raise TypeError("The expected type of `decision_vector` is `np.ndarray`")

        # P, C, and initial graph contain '-1'
        # self.initial_graph[np.where(self.initial_graph == -1)] = 0
        # self.initial_graph = (self.initial_graph).T

        #initial graph is a cpdag rewrite!!!!!
        parameters = {}
        parameters['lambda_value'] = 2
        # Ges-score
        if function_id == 0:
            # mix_graph = deepcopy(self.initial_graph_hc)
            # initial_pdag = WeightMatrix_Con(mix_graph.graph.copy(), self.constraint_matrix,
            #                                 decision_vector)  # delete the edges that been constrained
            # mix_pdag = np.matmul(self.pc_matrix, decision_vector) + initial_pdag
            # mix_graph_new = Matrix2Graph(mix_pdag, mix_graph)  # update the edges in mix_graph based on mix_pdag
            # # print(self.pc_matrix[:,:,0])
            # mix_graph2, illegal = PDAG2DAG_NonC(mix_graph_new, np.sum(self.pc_matrix, axis=2))  # mix_pdag may do not admit a consistent extension
            # if illegal:
            #     score = float('-inf')
            # else:
            #     score = GetBicScoreOfGraph(mix_graph2, self.data, False)
            # print('current min GesScore: ' + str(score))
            # return -score
            nowgraph = deepcopy(self.initial_graph)
            # score = GetBicScoreOfGraph(pdag2dag(self.initial_graph_hc), self.data, False)
            # print('HCScore:' + str(score))
            nodes = nowgraph.get_nodes()
            # print(decision_vector)
            ### use edge-oprator version:
            # for i in range(self.n):
            #     if decision_vector[i] == 1:
            #         fromnode = self.constraint_matrix[i, 0]
            #         tonode = self.constraint_matrix[i, 1]
            #         # if nowgraph.dpath[fromnode, tonode] == 1:
            #         #     sign = False
            #         #     score = float('-inf')
            #         #     break
            #         if nowgraph.graph[fromnode, tonode] == -1 and nowgraph.graph[tonode, fromnode] == 1:
            #             continue
            #         elif nowgraph.graph[tonode, fromnode] == -1 and nowgraph.graph[fromnode, tonode] == 1:
            #             nowgraph.remove_connecting_edge(nodes[fromnode], nodes[tonode])
            #             nowgraph.add_directed_edge(nodes[fromnode], nodes[tonode])
            #         elif nowgraph.graph[tonode, fromnode] == -1 and nowgraph.graph[fromnode, tonode] == -1:
            #             nowgraph.remove_connecting_edge(nodes[fromnode], nodes[tonode])
            #             nowgraph.add_directed_edge(nodes[fromnode], nodes[tonode])
            #         else:
            #             nowgraph.add_directed_edge(nodes[fromnode], nodes[tonode])
            # nowgraph.reconstitute_dpath(nowgraph.get_graph_edges()) ## need update the dpath-info
            # nowgraph_dag, illegal = PDAG2DAG_NonC(nowgraph, np.sum(self.pc_matrix, axis=2))

            ### use edge matrix version to add correct edges:
            initial_pdag = WeightMatrix_Con(nowgraph.graph.copy(), self.constraint_matrix,
                                            decision_vector)  # delete the edges that been constrained
            mix_pdag = np.matmul(self.pc_matrix, decision_vector) + initial_pdag
            nowgraph = Matrix2Graph(mix_pdag, nowgraph)  # update the edges in mix_graph based on mix_pdag

            ## if need delete the incorrect edges
            std = np.matmul(self.pc_matrix, 1 - decision_vector)
            std2 = np.trunc((np.abs(std) * nowgraph.graph - std)/2)
            std3 = WeightMatrix_Con(nowgraph.graph.copy(), self.constraint_matrix,
                                            1 - decision_vector)  # delete the edges that been constrained
            nowgraph = Matrix2Graph(np.sign(std2 - std2.T) + std3, nowgraph)

            # for i in range(self.n):
            #     if decision_vector[i] == 0:
            #         fromnode = self.constraint_matrix[i, 0]
            #         tonode = self.constraint_matrix[i, 1]
            #         if nowgraph.graph[fromnode, tonode] == -1 and nowgraph.graph[tonode, fromnode] == 1:
            #             nowgraph.remove_connecting_edge(nodes[fromnode], nodes[tonode])
            #         if nowgraph.graph[fromnode, tonode] == -1 and nowgraph.graph[tonode, fromnode] == -1:
            #             nowgraph.remove_connecting_edge(nodes[fromnode], nodes[tonode])
            #             nowgraph.add_directed_edge(nodes[tonode], nodes[fromnode])
            #         # if nowgraph.graph[fromnode, tonode] == 1 and nowgraph.graph[tonode, fromnode] == -1:
            #         #     nowgraph.remove_connecting_edge(nodes[fromnode], nodes[tonode])

            # if contain cycle
            std4 = deepcopy(nowgraph.graph)
            std5 = ((np.abs(std4) + std4).T)/2
            std6 = np.trace(expm(std5)) - self.m

            if std6 != 0:
                score = float('-inf')
            else:
                start_time =time.time()
                nowgraph_dag, illegal = PDAG2DAG_NonC(nowgraph, np.sum(self.pc_matrix, axis=2))
                end_time = time.time()
                self.time_NonC += end_time - start_time
                self.repeats_NonC += 1
                if illegal:
                    # raise TypeError("Have illegal!`")
                    score = float('-inf')
                else:
            # if illegal:
            #     score = float('-inf')
            # else:
                    score = GetBicScoreOfGraph(nowgraph_dag, self.data, False)
                # score = GetCorScoreOfGraph(nowgraph, self.data)
                # score = GetLogLikeScoreOfGraph(nowgraph, self.data)
            # print('current min ADDScore: ' + str(score))
            return -score

            # ud = np.sum(np.absolute(mix_graph2.graph))
            # return ud
        # if function_id == 1:
        #
        #     nowgraph = deepcopy(self.initial_graph_pc)
        #     # score = GetBicScoreOfGraph(pdag2dag(self.initial_graph_hc), self.data, False)
        #     # print('HCScore:' + str(score))
        #     # nodes = nowgraph.get_nodes()
        #     # print(decision_vector)
        #
        #
        #     ### use edge matrix version:
        #     decision_vector = np.ones(self.n) - decision_vector
        #     initial_pdag = WeightMatrix_Con(nowgraph.graph.copy(), self.constraint_matrix,
        #                                     decision_vector)  # delete the edges that been constrained
        #     mix_pdag = np.matmul(self.pc_matrix, decision_vector) + initial_pdag
        #     nowgraph = Matrix2Graph(mix_pdag, nowgraph)  # update the edges in mix_graph based on mix_pdag
        #     nowgraph_dag, illegal = PDAG2DAG_NonC(nowgraph, np.sum(self.pc_matrix, axis=2))
        #     if illegal:
        #         score = float('-inf')
        #     else:
        #         score = GetBicScoreOfGraph(nowgraph_dag, self.data, False)
        #         # score = GetCorScoreOfGraph(nowgraph_dag, self.data)
        #     print('current min EmptyScore: ' + str(score))
        #     return score


        # if function_id == 1:
        #     mix_graph = deepcopy(self.initial_graph_pc)
        #     nodes = mix_graph.get_nodes()
        #     for i in range(self.n):
        #         if decision_vector[i] == 0:
        #             fromnode = self.constraint_matrix[i, 0]
        #             tonode = self.constraint_matrix[i, 1]
        #             if mix_graph.graph[fromnode, tonode] == -1 and mix_graph.graph[tonode, fromnode] == 1:
        #                 mix_graph.remove_connecting_edge(nodes[fromnode], nodes[tonode])
        #             if mix_graph.graph[fromnode, tonode] == -1 and mix_graph.graph[tonode, fromnode] == -1:
        #                 mix_graph.remove_connecting_edge(nodes[fromnode], nodes[tonode])
        #             if mix_graph.graph[fromnode, tonode] == 1 and mix_graph.graph[tonode, fromnode] == -1:
        #                 mix_graph.remove_connecting_edge(nodes[fromnode], nodes[tonode])
        #     mix_graph2 = pdag2dag(mix_graph)
        #     score = GetBicScoreOfGraph(mix_graph2, self.data, False)
        #     # score = GetCorScoreOfGraph(mix_graph2, self.data)
        #     # iii = deepcopy(self.initial_graph_pc)
        #     # isd = InitialG2FinalG(decision_vector, iii, self.constraint_matrix, self.pc_matrix)
        #
        #     print('current min DECscore: ' + str(score))
        #     return -score
            # return -1


        if function_id == 1:
            nmi = 0

            # knowledge based data
            for i in range(len(decision_vector)):
                if decision_vector[i] == 1:
                    index1 = self.constraint_matrix[i, 0]
                    index2 = self.constraint_matrix[i, 1]
                    nmi += normalized_mutual_info_score(self.data.iloc[:, index1],  self.data.iloc[:, index2])
            if sum(decision_vector) == 0:
                avgnmi = 0
            else:
                avgnmi = nmi / sum(decision_vector)
            # print('avgnmi:' + str(avgnmi))

            ## knowledge true/false
            # for i in range(len(decision_vector)):
            #     if decision_vector[i] == 1:
            #         if i <= len(decision_vector)/2:
            #             nmi += 1.0
            #         else:
            #             nmi -= 1.0
            #     else:
            #         if i <= len(decision_vector)/2:
            #             nmi -= 1.0
            #         else:
            #             nmi += 1.0
            # if sum(decision_vector) == 0:
            #     avgnmi = 0
            # else:
            #     avgnmi = nmi / sum(decision_vector)
            # print('avgnmi:' + str(avgnmi))

            return - nmi



    def generate_random_solution(self):
        return self.evaluate(x=np.random.randint(0, 2, self.n).tolist()[:])












