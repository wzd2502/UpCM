import pandas as pd
from moead_framework.aggregation import Tchebycheff
from moead_framework.algorithm.combinatorial import Moead
from moead_framework.tool.result import save_population_full
from MOEAD import CausalGraphObject
from MOEAD import Problem
from MOEAD.Problem import CD_MOEAD_Problem
from MOEAD.CompareAndPlot import subplotnx,plotnx
from MOEAD.utilities import *
from MOEAD.InitialGraphs import *
from MOEAD.CompareAndPlot import *
from MOEAD.DataProcess import *
from MOEAD.PDAG2DAG_1 import pdag2dag_1
import os
import random
import time
import json

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
    repeat = kwargs.get('repeat', 0)
    selects = kwargs.get('select', ['pc'])

    cur_name = f'{network}_N{numofdata}_C{n_corcon}_W{n_wrocon}_{repeat}.csv'

    # load benchmark structure of network from 'os.path() + Datasets'
    dataset = pd.read_csv(os.path.join(rootpath, 'Datasets', network, 'Data_' + cur_name), header=0)
    dataset = dataset[:500]

    # load benchmark structure of network from 'os.path() + Networks'
    CGO1 = CausalGraphObject.CausalGraphObject(network)
    # rearrange the order of columes in dataset
    dataset = dataset[CGO1.benchmark_graph.get_node_names()]
    InitialGraphs = {}

    # load constraints from 'os.path() + Constraints'
    constraint_matrix = CGO1.read_constraints(cur_name=cur_name)
    CGO1.GetNMIofConstraints(dataset, constraint_matrix)
    pc_matrix = CGO1.get_pc_constraint_matrix(constraint_matrix)
    # write the constraints to file: os.path + Constraints

    # load basic and con graph of network from 'os.path() + Networks + Methods' (PC_stable, GES, MAHC, SaiyanH)
    print('GetInitializationPCGraph!')
    InitialGraphs['pc'] = GetPCGraph(dataset, CGO1.benchmark_graph)
    if 'pc' in selects:
        start_time_pc = time.time()
        InitialGraphs['pc_con'] = GetPCGraph_Con(dataset, CGO1.benchmark_graph, constraint_matrix)
        end_time_pc = time.time()
        pc_time = end_time_pc - start_time_pc
    if 'ges' in selects:
        #InitialGraphs['ges'] = GetLearnedGraph(network, method = 'GES', cur_name=cur_name, hasconstraints=False, example_CL=CGO1.benchmark_graph)
        InitialGraphs['ges_con'] = GetLearnedGraph(network, method='GES', cur_name=cur_name, hasconstraints=True,
                                               example_CL=CGO1.benchmark_graph)
    if 'mahc' in selects:
        #InitialGraphs['mahc'] = GetLearnedGraph(network, method = 'MAHC', cur_name=cur_name, hasconstraints=False, example_CL=CGO1.benchmark_graph)
        InitialGraphs['mahc_con'] = GetLearnedGraph(network, method='MAHC', cur_name=cur_name, hasconstraints=True,
                                                    example_CL=CGO1.benchmark_graph)
    if 'SaiyanH' in selects:
        #InitialGraphs['SaiyanH'] = GetLearnedGraph(network, method = 'SaiyanH', cur_name=cur_name, hasconstraints=False, example_CL=CGO1.benchmark_graph)
        InitialGraphs['SaiyanH_con'] = GetLearnedGraph(network, method='SaiyanH', cur_name=cur_name,
                                                       hasconstraints=True, example_CL=CGO1.benchmark_graph)
    InitialGraphs['empty'] = GetEmptyGraph(dataset, CGO1.benchmark_graph)

    # Define of MOP1
    results = pd.DataFrame(columns=['Network', 'Method', 'NumOfData', 'CorRate', 'WroRate', 'Repeat', 'SHD',
                                    'Adj_Tp', 'Adj_Fp', 'Adj_Tn', 'Adj_Fn', 'Prec', 'Rec', 'F1',
                                    'Con_Tp', 'Con_Fp', 'Con_Tn', 'Con_Fn', 'Con_Prec', 'Con_Rec', 'Con_F1'
                                    , 'type', 'Main_time', 'PDAG2DAG_Time', 'N_evaluations'])

    selectalg = 'pc'
    algname = 'UpCM'
    selectnet = InitialGraphs[selectalg]
    MOP1 = CD_MOEAD_Problem(num_of_objective= 2, constraint_matrix=constraint_matrix, InitialGraph = selectnet,
                                    data = dataset, pc_matrix = pc_matrix)

    ### MOEAD:
    #weight_file = os.getcwd() + '/Weights/' + str(MOP1.number_of_objective) + 'objs-' + str(number_of_weight) + 'wei.ws'
    # it is random sampled, 100*2
    weight_file = os.getcwd() + '/Weights/' + 'weights1.ws'

    start_time = time.time()  # 记录开始时间

    ### Update the max iteration of UpCM: wzd, 2025.2.24, revision 1

    n_evaluations = min(1000, 2**constraint_matrix.shape[0])
    
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

    save_file = os.path.join(rootpath, 'TrainingProcess', algname + '_' + cur_name \
                + "-W" + str(n_weights) \
                + "-T" + str(n_neighbors) \
                + "-iter" + str(n_evaluations) \
                + ".txt")

    end_time = time.time()  # 记录结束时间
    print(f"UpCM took {end_time - start_time:.4f} seconds.")
    print(f"PDAG2DAG_NonC took {MOP1.time_NonC:.4f} seconds.")
    UpCM_time = end_time - start_time
    PDAG2DAG_Time = MOP1.time_NonC
    
    save_population_full(save_file, population)

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
    bestscore = float('-inf')

    benchmark_graph = deepcopy(CGO1.benchmark_graph)

    for solution_object in population:
        pareto_front.append(solution_object.F)
        pareto_set.append(solution_object.decision_vector)

        # delete the dags contain cycles
        if solution_object.F[0] > 1000000:
            continue
        print('score ' + str(solution_object.F[0]) + ', Sum of NMI' + str(solution_object.F[1]))
        print(solution_object.decision_vector)

        #convert pdag2dag according decision_vector
        current_graph, IsLegal= InitialG2FinalG(solution_object.decision_vector, selectnet, constraint_matrix, pc_matrix) # it is a dag
        if not IsLegal:
            continue
        shd = SHD(benchmark_graph, current_graph).get_shd()
        # shd = CompareSHD(benchmark_graph.graph,current_graph.graph)
        score = GetBicScoreOfGraph(current_graph, dataset, False)
        graphacc = ComparePre_Rec(benchmark_graph, current_graph)
        Conacc = CompareConstraints_Rec(current_graph, constraint_matrix)

        print('Method: ' + algname + ', Score: ' + str(score) + ', SHD: ' + str(shd) + ', adjTp: ' + str(graphacc['Tp']) + ', adjFp: ' + str(graphacc['Fp']) + ', adjFn: ' + str(graphacc['Fn']) + ', adjTn: ' + str(graphacc['Tn']) +
              ', adjPrec: ' + str(graphacc['Prec']) + ', adjRec: ' + str(graphacc['Rec']) + ', F1: ' + str(graphacc['f1']) + ', ConPrec: ' + str(Conacc['Precision']) + ', ConRec: ' + str(Conacc['Recall']) + ', Con_F1: ' + str(Conacc['f1']))

        if shd <= minshd:
            if graphacc['f1'] >= maxf1:
                if Conacc['f1'] >= maxf1_con:
                    minshd = shd
                    maxf1 = graphacc['f1']
                    maxf1_con = Conacc['f1']
                    bestgraphacc = graphacc.copy()
                    bestConacc = Conacc.copy()
                    bestscore = score

        # only for draw the parto-front
        # newdata = {'Network': network, 'Method': 'UpCM' + str(time),  'SHD': str(shd),
        #            'Adj_Tp': graphacc['Tp'], 'Adj_Fp': graphacc['Fp'], 'Adj_Tn': graphacc['Tn'], 'Adj_Fn': graphacc['Fn'], 'Prec': graphacc['Prec'], 'Rec': graphacc['Rec'], 'F1': graphacc['f1'],
        #            'Con_Tp': Conacc['Tp'], 'Con_Fp': Conacc['Fp'], 'Con_Tn': Conacc['Tn'], 'Con_Fn': Conacc['Fn'], 'Con_Prec': Conacc['Precision'], 'Con_Rec': Conacc['Recall'], 'Con_F1': Conacc['f1'],
        #            'score': -1 * solution_object.F[0], 'avgnmi': -1 * solution_object.F[1]}
        # results = results.append(newdata, ignore_index=True)
        # time += 1
    newdata = {'Network': network, 'Method': algname, 'NumOfData': numofdata, 'CorRate': n_corcon,
               'WroRate': n_wrocon, 'Repeat': bestscore, 'SHD': minshd,
               'Adj_Tp': bestgraphacc['Tp'], 'Adj_Fp': bestgraphacc['Fp'], 'Adj_Tn': bestgraphacc['Tn'], 'Adj_Fn': bestgraphacc['Fn'], 'Prec': bestgraphacc['Prec'], 'Rec': bestgraphacc['Rec'], 'F1': bestgraphacc['f1'],
               'Con_Tp': bestConacc['Tp'], 'Con_Fp': bestConacc['Fp'], 'Con_Tn': bestConacc['Tn'], 'Con_Fn': bestConacc['Fn'], 'Con_Prec': bestConacc['Precision'], 'Con_Rec': bestConacc['Recall'], 'Con_F1': bestConacc['f1'],
               'type': 'Normal', 'Main_time': UpCM_time,  'PDAG2DAG_Time':PDAG2DAG_Time, 'N_evaluations':n_evaluations}
    new_row_df = pd.DataFrame([newdata])
    results = pd.concat([results, new_row_df], ignore_index=True)

    for comparealg in selects:
        # compared with orignal con-method
        key = f'{comparealg}_con'
        # key = 'pc'
        if key == 'pc_con' or key == 'pc':
            # let pc_stable satisfed constraints as possible, if pc_stable not constain cycles:
            current_graph = preorientate(InitialGraphs[key], constraint_matrix)
            current_graph, sign = pdag2dag_1(current_graph)
            # if pc_stable constain cycles, use standed pdag2dag
            if not sign:
                current_graph = pdag2dag(InitialGraphs[key])
        else:
            current_graph = InitialGraphs[key]

        shd = SHD(benchmark_graph, current_graph).get_shd()
        # shd = CompareSHD(benchmark_graph.graph, current_graph.graph)

        if acyclic(current_graph.graph):
            score = GetBicScoreOfGraph(current_graph, dataset, False)
        else:
            score = float('-inf')
        graphacc = ComparePre_Rec(benchmark_graph, current_graph)
        Conacc = CompareConstraints_Rec(current_graph, constraint_matrix)

        print('Method: ' + key + ', Score: ' + str(score) + ', SHD: ' + str(shd) + ', adjTp: ' + str(graphacc['Tp']) + ', adjFp: ' + str(graphacc['Fp']) + ', adjFn: ' + str(graphacc['Fn']) + ', adjTn: ' + str(graphacc['Tn']) +
              ', adjPrec: ' + str(graphacc['Prec']) + ', adjRec: ' + str(graphacc['Rec']) + ', F1: ' + str(graphacc['f1']) + ', ConPrec: ' + str(Conacc['Precision']) + ', ConRec: ' + str(Conacc['Recall'] ) + ', Con_F1: ' + str(Conacc['f1']))

        newdata = {'Network': network, 'Method': key,  'NumOfData': numofdata, 'CorRate': n_corcon,
                   'WroRate': n_wrocon, 'Repeat': score,
                   'SHD': shd,
                   'Adj_Tp': graphacc['Tp'], 'Adj_Fp': graphacc['Fp'], 'Adj_Tn': graphacc['Tn'],
                   'Adj_Fn': graphacc['Fn'], 'Prec': graphacc['Prec'], 'Rec': graphacc['Rec'],
                   'F1': graphacc['f1'],
                   'Con_Tp': Conacc['Tp'], 'Con_Fp': Conacc['Fp'], 'Con_Tn': Conacc['Tn'],
                   'Con_Fn': Conacc['Fn'], 'Con_Prec': Conacc['Precision'], 'Con_Rec': Conacc['Recall'],
                   'Con_F1': Conacc['f1'], 'Main_time': pc_time
                   }
        new_row_df = pd.DataFrame([newdata])
        results = pd.concat([results, new_row_df], ignore_index=True)

    ## only for draw the parto-front
    # resultname = 'Scatter_' + cur_name
    # results.to_csv(os.getcwd() +  f'/Results/{network}/{resultname}')
    # PlotScatter(results)

    ## when execute
    resultname = f'results_{cur_name}'
    results.to_csv(os.getcwd() +  f'/RevisionResults/{network}/{resultname}')

    return results


def generatedataandconstraints(kwargs):

    network = kwargs.get('network', 'Alarm')
    numofdata = kwargs.get('numofdata', 100)
    n_corcon = kwargs.get('n_corcon', 0.2)
    n_wrocon = kwargs.get('n_wrocon', 1)
    repeat = kwargs.get('repeat', 0)

    cur_name = f'{network}_N{numofdata}_C{n_corcon}_W{n_wrocon}_{repeat}'
    BayesToolpath = 'D:\\code_other\\bayesys_system\\Bayesys_Release_v3.6\\Input\\Multiple inputs'

    dataset = csv2dataset(network, numofdata)
    dataset.to_csv(os.path.join(rootpath, 'Datasets', network, 'Data_' + cur_name + '.csv'), index = False)
    # also cc to Bayes Toolboxs
    dataset.to_csv(os.path.join(BayesToolpath, 'Data_' + cur_name + '.csv'), index = False)

    CGO1 = CausalGraphObject.CausalGraphObject(network)
    constraint_matrix = CGO1.initial_constraints(n_wrocon, n_corcon)
    constraint_pd = CGO1.WriteConstraintsTocsv(constraint_matrix, cur_name)
    # also cc to Bayes Toolboxs
    constraint_pd.to_csv(os.path.join(BayesToolpath, 'Constraints_' + cur_name + '.csv'), index = False)

    ###update the settting files in BayesTool:
    settings = pd.read_csv(os.path.join(BayesToolpath, 'settings.csv'))
    for algname in ['MAHC', 'GES', 'SaiyanH']:
        new_row = {'Algorithm': algname,
                   'trainingDataFileName': 'Data_' + cur_name,
                   'DAGtrueFileName': f'DAGtrue_{network}',
                   'directedConstraintsFileName': 'FALSE',
                   'undirectedConstraintsFileName': 'FALSE',
                   'forbiddenConstraintsFileName': 'FALSE',
                   'temporalConstraintsFileName': 'FALSE',
                   'strictTemporal': 'FALSE',
                   'initialGraphConstraintsFileName': 'Constraints_' + cur_name,
                   'varRelevantConstraints': 'FALSE',
                   'Pruning': 3,
                   'BIC': 2}
        settings = pd.concat([settings, pd.DataFrame([new_row])], ignore_index=True)
    settings.to_csv(os.path.join(BayesToolpath, 'settings.csv'), index = False)

if __name__ == '__main__':

    mode = 'Compare'
    if mode == 'Generate':
        # for network in ['ASIA', 'SPORTS', 'DIARRHOEA', 'PROPERTY', 'ALARM', 'FORMED']:
        for network in ['Large-SF100', 'Large-SF200', 'Large-SF500','Large-ER100','Large-ER200', 'Large-ER500', ]:
            for n_wrocon in [1]:
                for repeat in range(10):
                    parameters = {
                        'network': network,
                        'numofdata': 500,
                        'n_corcon': 0.2,
                        'n_wrocon': 1,
                        'repeat': repeat,
                        'select': []
                    }
                    generatedataandconstraints(parameters)


    if mode == 'Compare':

        # # For synthetic dataset
        # # for network in ['ASIA', 'SPORTS', 'DIARRHOEA', 'PROPERTY', 'ALARM', 'FORMED']:
        # # for network in ['DIARRHOEA', 'PROPERTY', 'ALARM', 'FORMED']:
        # for network in ['Large-SF100','Large-ER100', 'Large-SF200','Large-ER200']:
        # #     for n_evaluations in [10, 100, 500, 1000]:
        #     # for n_corcon in [0.2]:
        #         Time_used = {'UpCM': 0, 'PC_con': 0, 'PDAG2DAG':0, 'Repeat_times': 0}
        #         for repeat in range(10):
        #             print(f'{repeat}-th UpCM for {network}:')
        #         # for repeat in []:
        #             parameters = {
        #                 'network': network,
        #                 'numofdata': 500,
        #                 'n_corcon': 0.2,
        #                 'n_wrocon': 1,
        #                 'n_weights': 100,
        #                 'n_neighbors': 100,
        #                 'n_evaluations': 1000,
        #                 'repeat': repeat,
        #                 'select': ['pc']
        #             }
        #             results = UCDForMOEAD(parameters)
        #             Time_used['UpCM'] += results.loc[0, 'Main_time']
        #             Time_used['PC_con'] += results.loc[1, 'Main_time']
        #             Time_used['PDAG2DAG'] += results.loc[0, 'PDAG2DAG_Time']
        #             Time_used['Repeat_times'] += results.loc[0, 'N_evaluations']
        #         Time_used = {key: value / 10 for key, value in Time_used.items()}
        #         with open(f'RevisionResults\\TimeUsed_{network}.json', 'w') as file:
        #             json.dump(Time_used, file)

        #For real dataset
        parameters = {
            'network': 'COVID19',
            'numofdata': 866,
            'n_corcon': 0.2,
            'n_wrocon': 1,
            'n_weights': 100,
            'n_neighbors': 100,
            'n_evaluations': 1000,
            'repeat': 0,
            'select': ['pc', 'ges','mahc','SaiyanH']
            }
        UCDForMOEAD(parameters)

    if mode == 'Plot':
        filepath =os.path.join(rootpath, 'Final_Results.xlsx')
        PlotResults(filepath, mode = 'n_wrocon')

        ### n_evaluations
        # filepath = os.path.join(rootpath, 'Para_Experiment', 'NumofEvaluations', 'Final_Results.xlsx')
        # PlotResults(filepath, mode = 'n_evaluations')

        ### n_weights
        # filepath = os.path.join(rootpath, 'Para_Experiment', 'Weights', 'Final_Results.xlsx')
        # PlotResults(filepath, mode = 'weights')


        ### scatter:
        # results = pd.read_csv(os.path.join(rootpath,'Results\PROPERTY', 'Scatter_PROPERTY_N100_C0.2_W1_8.csv'), header=0)
        # PlotScatter(results)



