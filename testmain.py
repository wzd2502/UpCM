import pandas as pd
from moead_framework.aggregation import Tchebycheff
from moead_framework.algorithm.combinatorial import Moead
from moead_framework.tool.result import save_population_full
from matplotlib import pyplot as plt


from causallearn.utils.Dataset import load_dataset
from causallearn.utils.TXT2GeneralGraph import txt2generalgraph
import os
import numpy as np
from copy import deepcopy

from causallearn.utils.PDAG2DAG import pdag2dag
from causallearn.utils.DAG2CPDAG import dag2cpdag
from causallearn.graph.SHD import SHD
from causallearn.graph.AdjacencyConfusion import AdjacencyConfusion
from causallearn.score.LocalScoreFunction import local_score_BIC
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge

from MOEAD import CausalGraphObject
from MOEAD import Problem
from MOEAD.Problem import CD_MOEAD_Problem
from MOEAD.CompareAndPlot import subplotnx,plotnx
from MOEAD.utilities import *
from MOEAD.InitialGraphs import *
from MOEAD.CompareAndPlot import *
from MOEAD.DataProcess import *
import os
import random

random.seed(52)
rootpath = os.getcwd()

def UCDForMOEAD(kwargs):

    '''
    :param kwargs:
    :return:

    '''
    network = kwargs.get('network', 'Alarm')
    numofdata = kwargs.get('numofdata', 100)
    n_corcon = kwargs.get('n_corcon', 0.2)
    n_wrocon = kwargs.get('n_wrocon', 1)
    n_weights = kwargs.get('n_weights', 100)
    n_neighbors = kwargs.get('n_neighbors', 100)
    n_evaluations = kwargs.get('n_evaluations', 1000)
    repeat = kwargs.get('repeat', 3)


    # load dataset + benchmark structure of network from 'os.path() + Datasets'
    # also write dataset to file: os.path+ Datasets\Samples.
    dataset = csv2dataset(network, numofdata, True, repeat)
    CGO1 = CausalGraphObject.CausalGraphObject(network)
    dataset = dataset[CGO1.benchmark_graph.get_node_names()]
    InitialGraphs = {}

    # PC-stable and GES
    InitialGraphs['pc'] = GetPCGraph(dataset, CGO1.benchmark_graph)
    InitialGraphs['ges'] = GetGesGraph(network, repeat, False, CGO1.benchmark_graph)
    #InitialGraphs['hc'] = GetHCGraph(dataset, CGO1.benchmark_graph)
    InitialGraphs['empty'] = GetEmptyGraph(dataset, CGO1.benchmark_graph)

    # get constraints including wrong and right
    constraint_matrix = CGO1.initial_constraints(n_wrocon, n_corcon)
    CGO1.GetNMIofConstraints(dataset, constraint_matrix)
    pc_matrix = CGO1.get_pc_constraint_matrix(constraint_matrix)
    # write the constraints to file: os.path + Constraints

    # other compare algorithms, including PC_stable(add priors), MAHC, SaiyanH, GES, HC(added by Bayes)
    InitialGraphs['pc_con'] = GetPCGraph_Con(dataset, CGO1.benchmark_graph, constraint_matrix)
    InitialGraphs['ges_con'] = GetGesGraph(network, repeat, True, CGO1.benchmark_graph)
    # InitialGraphs['mahc'] = GetMahcGraph(network, CGO1.benchmark_graph)

    # Define of MOP1
    MOP1 = CD_MOEAD_Problem(num_of_objective= 2, constraint_matrix=constraint_matrix, InitialGraphs = InitialGraphs,
                                        data = dataset, pc_matrix = pc_matrix)

    ### MOEAD:
    #weight_file = os.getcwd() + '/Weights/' + str(MOP1.number_of_objective) + 'objs-' + str(number_of_weight) + 'wei.ws'
    # it is random sampled, 100*2
    weight_file = os.getcwd() + '/Weights/' + 'weights.ws'
    ###############################
    #    Execute the algorithm    #
    ###############################
    moead = Moead(problem=MOP1,
                    max_evaluation=n_evaluations,
                    number_of_weight_neighborhood=n_neighbors,
                    weight_file=weight_file,
                    aggregation_function=Tchebycheff,
                    )

    population = moead.run()

    ###############################
    #    Extract the Pareto set   #
    #     and the Pareto front    #
    ###############################
    pareto_front = []
    pareto_set = []

    minshd = 1000
    maxf1 = 0
    maxf1_con = 0
    bestgraphacc = {}
    bestConacc = {}

    benchmark_graph = deepcopy(CGO1.benchmark_graph)
    time = 0
    results = pd.DataFrame(columns=['Network', 'Method', 'NumOfData', 'CorRate', 'WroRate', 'Repeat', 'SHD',
                                    'Adj_Tp', 'Adj_Fp', 'Adj_Tn', 'Adj_Fn', 'Prec', 'Rec', 'F1',
                                    'Con_Tp', 'Con_Fp', 'Con_Tn', 'Con_Fn', 'Con_Prec', 'Con_Rec', 'Con_F1'
                                    ])

    for solution_object in population:
        pareto_front.append(solution_object.F)
        pareto_set.append(solution_object.decision_vector)

        # delete the dags contain cycles
        if solution_object.F[0] > 1000000:
            continue
        print('score ' + str(solution_object.F[0]) + ', Sum of NMI' + str(solution_object.F[1]))
        print(solution_object.decision_vector)

        current_graph= InitialG2FinalG(solution_object.decision_vector, InitialGraphs['ges'], constraint_matrix, pc_matrix, type= 'ges') # it is a dag
        shd = SHD(benchmark_graph, current_graph).get_shd()
        # shd = CompareSHD(benchmark_graph.graph,current_graph.graph)
        score = GetBicScoreOfGraph(current_graph, dataset, False)
        graphacc = ComparePre_Rec(benchmark_graph, current_graph)
        Conacc = CompareConstraints_Rec(current_graph, constraint_matrix)

        print('Method: My_1, Score: ' + str(score) + ', SHD: ' + str(shd) + ', adjTp: ' + str(graphacc['Tp']) + ', adjFp: ' + str(graphacc['Fp']) + ', adjFn: ' + str(graphacc['Fn']) + ', adjTn: ' + str(graphacc['Tn']) +
              ', adjPrec: ' + str(graphacc['Prec']) + ', adjRec: ' + str(graphacc['Rec']) + ', F1: ' + str(graphacc['f1']) + ', ConPrec: ' + str(Conacc['Precision']) + ', ConRec: ' + str(Conacc['Recall']) + ', Con_F1: ' + str(Conacc['f1']))
        if shd <= minshd:
            if graphacc['f1'] >= maxf1:
                if Conacc['f1'] >= maxf1_con:
                    minshd = shd
                    maxf1 = graphacc['f1']
                    maxf1_con = Conacc['f1']
                    bestgraphacc = graphacc.copy()
                    bestConacc = Conacc.copy()

        # newdata = {'Network': network, 'Method': 'MOEAD_CD' + str(time),  'SHD': str(shd),
        #            'Adj_Tp': graphacc['Tp'], 'Adj_Fp': graphacc['Fp'], 'Adj_Tn': graphacc['Tn'], 'Adj_Fn': graphacc['Fn'], 'Prec': graphacc['Prec'], 'Rec': graphacc['Rec'], 'F1': graphacc['f1'],
        #            'Con_Tp': Conacc['Tp'], 'Con_Fp': Conacc['Fp'], 'Con_Tn': Conacc['Tn'], 'Con_Fn': Conacc['Fn'], 'Con_Prec': Conacc['Precision'], 'Con_Rec': Conacc['Recall'], 'Con_F1': Conacc['f1'],
        #            'score': -1 * solution_object.F[0], 'avgnmi': -1 * solution_object.F[1]}
        # results = results.append(newdata, ignore_index=True)
        # time += 1
    newdata = {'Network': network, 'Method': 'MOEAD_CD', 'NumOfData': numofdata, 'CorRate': n_corcon,
               'WroRate': n_wrocon, 'Repeat': repeat, 'SHD': minshd,
               'Adj_Tp': bestgraphacc['Tp'], 'Adj_Fp': bestgraphacc['Fp'], 'Adj_Tn': bestgraphacc['Tn'], 'Adj_Fn': bestgraphacc['Fn'], 'Prec': bestgraphacc['Prec'], 'Rec': bestgraphacc['Rec'], 'F1': bestgraphacc['f1'],
               'Con_Tp': bestConacc['Tp'], 'Con_Fp': bestConacc['Fp'], 'Con_Tn': bestConacc['Tn'], 'Con_Fn': bestConacc['Fn'], 'Con_Prec': bestConacc['Precision'], 'Con_Rec': bestConacc['Recall'], 'Con_F1': bestConacc['f1']}
    results = results.append(newdata, ignore_index=True)

    for key in InitialGraphs.keys():
        if key != 'empty':
            current_graph = InitialGraphs[key]
            if key == 'pc'  or key == 'pc_con':
                # let pc_stable satisfed constraints as possible
                current_graph = preorientate(current_graph, constraint_matrix)
                current_graph = pdag2dag(current_graph)

            shd = SHD(benchmark_graph, current_graph).get_shd()
            # shd = CompareSHD(benchmark_graph.graph, current_graph.graph)
            score = GetBicScoreOfGraph(current_graph, dataset, False)
            graphacc = ComparePre_Rec(benchmark_graph, current_graph)
            Conacc = CompareConstraints_Rec(current_graph, constraint_matrix)

            print('Method: ' + key + ', Score: ' + str(score) + ', SHD: ' + str(shd) + ', adjTp: ' + str(graphacc['Tp']) + ', adjFp: ' + str(graphacc['Fp']) + ', adjFn: ' + str(graphacc['Fn']) + ', adjTn: ' + str(graphacc['Tn']) +
                  ', adjPrec: ' + str(graphacc['Prec']) + ', adjRec: ' + str(graphacc['Rec']) + ', F1: ' + str(graphacc['f1']) + ', ConPrec: ' + str(Conacc['Precision']) + ', ConRec: ' + str(Conacc['Recall'] ) + ', Con_F1: ' + str(Conacc['f1']))

            newdata = {'Network': network, 'Method': key,  'NumOfData': numofdata, 'CorRate': n_corcon,
                       'WroRate': n_wrocon, 'Repeat': repeat,
                       'SHD': shd,
                       'Adj_Tp': graphacc['Tp'], 'Adj_Fp': graphacc['Fp'], 'Adj_Tn': graphacc['Tn'],
                       'Adj_Fn': graphacc['Fn'], 'Prec': graphacc['Prec'], 'Rec': graphacc['Rec'],
                       'F1': graphacc['f1'],
                       'Con_Tp': Conacc['Tp'], 'Con_Fp': Conacc['Fp'], 'Con_Tn': Conacc['Tn'],
                       'Con_Fn': Conacc['Fn'], 'Con_Prec': Conacc['Precision'], 'Con_Rec': Conacc['Recall'],
                       'Con_F1': Conacc['f1']
                       }
            results = results.append(newdata, ignore_index=True)

        score_benchmark = GetBicScoreOfGraph(benchmark_graph, dataset, False)
        print('benchmark score: ' + str(score_benchmark))

    # results = pd.read_csv(os.path.join(rootpath, 'Results', resultname))
    # PlotScatter(results)


if 1:

    parameters = {
        'network': 'ALARM',
        'numofdata': 100,
        'n_corcon': 0.2,
        'n_wrocon': 1,
        'n_weights': 100,
        'n_neighbors': 100,
        'n_evaluations': 1000
    }
    UCDForMOEAD(parameters)



