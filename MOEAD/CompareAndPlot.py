import os.path
from copy import deepcopy

import numpy as np

from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.Dag import Dag
from causallearn.utils.DAG2CPDAG import dag2cpdag
from MOEAD.utilities import PDAG2DAG_NonC
import networkx as nx
from matplotlib import pyplot as plt
from MOEAD.Metrics import EdgeConfusion
import pandas as pd
from matplotlib.ticker import MaxNLocator, FixedLocator
from MOEAD.DataProcess import csv2generalgraph
from causallearn.graph.SHD import SHD
import seaborn as sns
from causallearn.graph.AdjacencyConfusion import AdjacencyConfusion
from causallearn.graph.ArrowConfusion import ArrowConfusion
import csv, math
import matplotlib.ticker as ticker

plt.rcParams['font.family'] = 'Times New Roman'

def plotnx(decision_vector, initial_graph: GeneralGraph, constraint_matrix, pc_matrix):

    nodes = initial_graph.get_nodes()
    current_graph = deepcopy(initial_graph)
    for j in range(len(decision_vector)):
        if decision_vector[j] != 0:
            current_graph.add_directed_edge(nodes[int(constraint_matrix[j,0])], nodes[int(constraint_matrix[j,1])])
    current_graph = PDAG2DAG_NonC(current_graph, np.sum(pc_matrix, axis=2))

    subplotnx(current_graph, 'current_graph')

def subplotnx(G:GeneralGraph, name:str):
    g = nx.DiGraph()
    for edge in G.get_graph_edges():
        fromnode = edge.node1.get_name()
        tonode = edge.node2.get_name()
        g.add_edge(fromnode, tonode)

    plt.tight_layout()
    nx.draw_networkx(g, arrows=True)
    plt.title(name)
    plt.show()

def ComparePre_Rec(benchmarkgraph:GeneralGraph, learnedgraph:GeneralGraph) -> {}:

    result = {}
    adj = EdgeConfusion(benchmarkgraph, learnedgraph)

    result['Tp'] = adj.get_adj_tp()
    result['Fp'] = adj.get_adj_fp()
    result['Fn'] = adj.get_adj_fn()
    result['Tn'] = adj.get_adj_tn()

    result['Prec'] = adj.get_adj_precision()
    result['Rec'] = adj.get_adj_recall()

    # adj = ArrowConfusion(benchmarkgraph, learnedgraph)
    # result['Tp'] = adj.get_arrows_tp()
    # result['Fp'] = adj.get_arrows_fp()
    # result['Fn'] = adj.get_arrows_fn()
    # result['Tn'] = adj.get_arrows_tn()
    #
    # result['Prec'] = adj.get_arrows_precision()
    # result['Rec'] = adj.get_arrows_recall()


    if result['Prec'] + result['Rec'] == 0:
        result['f1'] = 0
    else:
        result['f1'] = 2 * result['Prec'] * result['Rec']/ (result['Prec'] + result['Rec'])

    return result

def CompareConstraints_Rec(learnedgraph:GeneralGraph, constraints_matrix:np.array([[]])) -> {}:

    Tp = 0
    Fp = 0
    Fn = 0
    Tn = 0
    for i in range(constraints_matrix.shape[0]):
        fromnode = constraints_matrix[i][0]
        tonode = constraints_matrix[i][1]
        right = constraints_matrix[i][3]
        if right:
            if learnedgraph.graph[fromnode,tonode] == -1 and learnedgraph.graph[tonode,fromnode] == 1:
                Tp += 1
            else:
                Fn += 1
        if not right:
            if learnedgraph.graph[fromnode,tonode] != -1 and learnedgraph.graph[tonode,fromnode] != 1:
                Tn += 1
            else:
                Fp += 1
    if Tp + Fp != 0:
        Precision = Tp / (Tp + Fp)
    else:
        Precision = 0

    if Tp + Fn != 0:
        Recall = Tp / (Tp + Fn)
    else:
        Recall = 0

    if Precision + Recall == 0:
        f1 = 0
    else:
        f1 = 2 * Precision * Recall / (Precision + Recall)

    return {'Tp': Tp, 'Fp': Fp, 'Fn': Fn, 'Tn': Tn, 'Precision': Precision, 'Recall': Recall, 'f1': f1}


def CompareSHD(G_true:np.array([[]]), G_est: np.array([[]])):

    #: two graphs are DAG
    shd = 0
    G_true =  np.abs(-1 * np.abs(G_true) + G_true)/2
    G_est = np.abs(-1 * np.abs(G_est) + G_est)/2
    for i in range(G_true.shape[0]):
        for j in range(i+1, G_true.shape[1]):
            if G_true[i,j] == 1:
                if G_est[i,j] == 0 or G_est[j,i] == 1:
                    shd += 1
            elif G_true[i,j] == 0 and G_est[i,j] == 1:
                shd += 1

    return shd

def PlotScatter(results):

    results_cdup = results[~results['Method'].isin(['pc', 'pc_con'])]
    results_pc = results[results['Method'] == 'pc_con']
    thresholds = {'SHD': int(results_pc['SHD']), 'F1': float(results_pc['F1']), 'F1_con': float(results_pc['Con_F1'])}
    data = {
        'score': results_cdup['score'],
        'avgnmi': results_cdup['avgnmi'],
        'SHD': results_cdup['SHD'],
        'F1': results_cdup['F1'],
        'F1_con': results_cdup['Con_F1'],
    }

    df = pd.DataFrame(data)
    df['SHD'] = df['SHD'].astype(int)
    df['F1'] = df['F1'].astype(float)
    df['F1_con'] = df['F1_con'].astype(float)

    # 创建三幅图
    fig, axs = plt.subplots(1, 3, figsize=(9, 3.4))
    colors = [(189 / 255, 60 / 255, 51 / 255), (249 / 255, 198 / 255, 118 / 255), (127 / 255, 169 / 255, 205 / 255)]
    # 遍历每个指标，绘制不同颜色的散点图
    for idx, (key, threshold) in enumerate(thresholds.items()):


        if key == 'SHD':
            ax = axs[idx]
            ax.xaxis.set_major_locator(MaxNLocator(4))
            ax.yaxis.set_major_locator(MaxNLocator(4))
            min_index = df[key].idxmin()
            ax.scatter(df[df[key] > threshold]['score'], df[df[key] > threshold]['avgnmi'], color=colors[0], label=f'Worse than PC_stable')
            ax.scatter(df[df[key] == threshold]['score'], df[df[key] == threshold]['avgnmi'], color=colors[1],
                       label=f'Equal to PC_stable')
            ax.scatter(df[df[key] < threshold]['score'], df[df[key] < threshold]['avgnmi'], color=colors[2],
                       label=f'Better than PC_stable')
            ax.annotate(f'lowest {key}', (df.iloc[min_index]['score'], df.iloc[min_index]['avgnmi']),
                         textcoords="offset points", xytext=(-20, 10), ha='center', fontsize = 10)
            if df.iloc[min_index][key] < threshold:
                col = colors[2]
            elif df.iloc[min_index][key] > threshold:
                col = colors[0]
            else:
                col = colors[1]
            ax.scatter(df.iloc[min_index]['score'], df.iloc[min_index]['avgnmi'], marker='*', s=200, color = colors[2])
            ax.set_xlabel('Score', fontsize=12)
            ax.set_ylabel('Knowledge (NMI)', fontsize=12)
            # ax.legend(fontsize=12)
            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.set_xlim(min(df['score']) - (max(df['score']) - min(df['score']) ) * 0.1,
                        max(df['score']) + (max(df['score']) - min(df['score']) ) * 0.1)
            ax.set_ylim(min(df['avgnmi']) - (max(df['avgnmi']) - min(df['avgnmi']) ) * 0.1,
                        max(df['avgnmi']) + + (max(df['avgnmi']) - min(df['avgnmi']) ) * 0.1)
            ax.set_title(f'UpCM vs PC_stable: {key}')

            # 自定义纵坐标刻度格式
            def format_func(value, tick_number):
                return f'{value:.1f}'

            # 设置纵坐标刻度格式
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_func))
        else:
            if key == 'F1_con':
                key1 = r'F1$^c$'
            else:
                key1 = key
            ax = axs[idx]
            ax.xaxis.set_major_locator(MaxNLocator(4))
            ax.yaxis.set_major_locator(MaxNLocator(4))
            min_index = df[key].idxmax()
            ax.scatter(df[df[key] < threshold]['score'], df[df[key] < threshold]['avgnmi'], color=colors[0], label=f'Worse than PC_stable')
            ax.scatter(df[df[key] == threshold]['score'], df[df[key] == threshold]['avgnmi'], color=colors[1],
                       label=f'Equal to PC_stable')
            ax.scatter(df[df[key] > threshold]['score'], df[df[key] > threshold]['avgnmi'], color=colors[2],
                       label=f'Better than PC_stable')
            ax.annotate(f'Highest {key1}', (df.iloc[min_index]['score'], df.iloc[min_index]['avgnmi']),
                         textcoords="offset points", xytext=(-20, 10), ha='center', fontsize = 10)
            if df.iloc[min_index][key] < threshold:
                col = colors[2]
            elif df.iloc[min_index][key] > threshold:
                col = colors[0]
            else:
                col = colors[1]
            ax.scatter(df.iloc[min_index]['score'], df.iloc[min_index]['avgnmi'], marker='*', s=200, color = colors[2])
            ax.set_xlabel('Score', fontsize=12)
            ax.set_ylabel('Knowledge (NMI)', fontsize=12)
            #ax.legend(fontsize=12)
            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.set_xlim(min(df['score']) - (max(df['score']) - min(df['score'])) * 0.1,
                        max(df['score']) + (max(df['score']) - min(df['score'])) * 0.1)
            ax.set_ylim(min(df['avgnmi']) - (max(df['avgnmi']) - min(df['avgnmi'])) * 0.1,
                        max(df['avgnmi']) + + (max(df['avgnmi']) - min(df['avgnmi'])) * 0.1)
            ax.set_title(f'UpCM vs PC_stable: {key1}')

            def format_func(value, tick_number):
                return f'{value:.1f}'

            # 设置纵坐标刻度格式
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_func))

    # plt.tight_layout()
    # #plt.savefig('ALARM_100_1_0.2.png', dpi=300)
    # plt.show()

    legend1 = plt.Line2D([], [], marker='o', color='w', markerfacecolor=colors[0], markersize=5, label='Worse than PC_stable')
    legend2 = plt.Line2D([], [], marker='o', color='w', markerfacecolor=colors[1], markersize=5, label='Equal to PC_stable')
    legend3 = plt.Line2D([], [], marker='o', color='w', markerfacecolor=colors[2], markersize=5, label='Better than PC_stable')
    fig.legend(handles = [legend1, legend2, legend3], loc='lower center', bbox_to_anchor=(0.5, 0),  ncol=4)
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    plt.show()


def CompareBayesMethods(filepath, constraintspath, network, numofdata, n_corcon, n_wrocon):

    results = pd.DataFrame(columns=['Network', 'Method', 'NumOfData', 'CorRate', 'WroRate', 'Repeat', 'SHD',
                                    'Adj_Tp', 'Adj_Fp', 'Adj_Tn', 'Adj_Fn', 'Prec', 'Rec', 'F1',
                                    'Con_Tp', 'Con_Fp', 'Con_Tn', 'Con_Fn', 'Con_Prec', 'Con_Rec', 'Con_F1'
                                    ])
    csv_file = filepath
    std_graph = csv2generalgraph(network)
    with open(csv_file, mode='r') as file:
        reader = csv.reader(file)

        repeat = -1
        for row in reader:
            if row[0] == 'ID':
                if repeat >= 0:
                    constraints = pd.read_csv(os.path.join(constraintspath, f'{network}_{math.floor(repeat/10)}.csv'), index_col=0)
                    conmatrix = np.zeros([constraints.shape[0],4], dtype=int)
                    for i in range(constraints.shape[0]):
                        constraint = constraints.iloc[i]
                        conmatrix[i, 0] = int(g.node_map[g.get_node(constraint[0])])
                        conmatrix[i, 1] = int(g.node_map[g.get_node(constraint[1])])
                        conmatrix[i, 2] = int(i)
                        conmatrix[i, 3] = int(constraint[2])

                    shd = SHD(std_graph, g).get_shd()
                    graphacc = ComparePre_Rec(std_graph, g)
                    Conacc = CompareConstraints_Rec(g, conmatrix)
                    if repeat <10:
                        method = 'HC'
                    elif repeat < 20:
                        method = 'GES'
                    elif repeat < 30:
                        method = 'MAHC'
                    else:
                        method = 'SaiyanH'
                    newdata = {'Network': network, 'Method': method, 'NumOfData': numofdata, 'CorRate': n_corcon,
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
                repeat +=1



                # generate new graph
                g = deepcopy(std_graph)
                g.remove_edges(g.get_graph_edges())
                continue
            else:
                fromnode = g.get_node(row[1])
                tonode = g.get_node(row[3])
                g.add_directed_edge(fromnode,tonode)

    results.to_csv('CompareBayesMethods.csv', index=False)

def PlotResults(filepath, mode):

    results = pd.read_excel(filepath)

    if mode == 'numofdata':
        plotdata = pd.DataFrame(columns=['Network', 'NumOfData', 'Method', 'Metric', 'Value'])

        condition1 = (results['CorRate'] == 0.2) & (results['WroRate'] == 1)
        results = results[condition1]
        # for network in ['ASIA', 'SPORTS', 'DIARRHOEA', 'PROPERTY', 'ALARM', 'FORMED']:
        for network in ['ALARM', 'FORMED']:
            for numofdata in [30, 50, 100, 200]:
                for alg in ['UpCM', 'PCstable_con', 'GES_con', 'MAHC_con']:
                    condition2 = (results['Network'] == network) & (results['NumOfData'] == numofdata) & (results['Method'] == alg) & (results['Repeat'] > -1000000) #delete -inf score
                    AvgSHD = results[condition2]['SHD'].mean()
                    AvgF1 = results[condition2]['F1'].mean()
                    AvgF1_con = results[condition2]['Con_F1'].mean()
                    if alg == 'UpCM':
                        alg1 = 'UpCM'
                    elif alg == 'PCstable_con':
                        alg1 = r'PCstable$^{\triangledown}$'
                    elif alg == 'GES_con':
                        alg1 = r'GES$^{\triangledown}$'
                    elif alg == 'MAHC_con':
                        alg1 = r'MAHC$^{\triangledown}$'
                    newdata1 = {'Network': network, 'NumOfData': numofdata, 'Method': alg1,
                               'Metric': 'SHD', 'Value': AvgSHD}
                    newdata2 = {'Network': network, 'NumOfData': numofdata, 'Method': alg1,
                               'Metric': 'F1', 'Value': AvgF1}
                    newdata3 = {'Network': network, 'NumOfData': numofdata, 'Method': alg1,
                               'Metric': r'F1$^c$', 'Value': AvgF1_con}
                    plotdata = plotdata.append(newdata1, ignore_index=True)
                    plotdata = plotdata.append(newdata2, ignore_index=True)
                    plotdata = plotdata.append(newdata3, ignore_index=True)

        support(plotdata, index1= 'Network', index2= 'Metric',index3 = 'NumOfData')
        plotdata.to_csv(os.getcwd() +  f'/Plots/NumofData/diarrhoeaandproperty2.csv')


    if mode == 'n_corcon':
        plotdata = pd.DataFrame(columns=['Network', 'CorRate', 'Method', 'Metric', 'Value'])

        condition1 = (results['NumOfData'] == 100) & (results['WroRate'] == 1)
        results = results[condition1]
        # for network in ['ASIA', 'SPORTS', 'DIARRHOEA', 'PROPERTY', 'ALARM', 'FORMED']:
        for network in ['ASIA', 'SPORTS']:
            for corcon in [0.1, 0.2, 0.5]:
                for alg in ['UpCM', 'PCstable_con', 'GES_con', 'MAHC_con']:
                    condition2 = (results['Network'] == network) & (results['CorRate'] == corcon) & (results['Method'] == alg) & (results['Repeat'] > -1000000) #delete -inf score
                    AvgSHD = results[condition2]['SHD'].mean()
                    AvgF1 = results[condition2]['F1'].mean()
                    AvgF1_con = results[condition2]['Con_F1'].mean()
                    if alg == 'UpCM':
                        alg1 = 'UpCM'
                    elif alg == 'PCstable_con':
                        alg1 = r'PCstable$^{\triangledown}$'
                    elif alg == 'GES_con':
                        alg1 = r'GES$^{\triangledown}$'
                    elif alg == 'MAHC_con':
                        alg1 = r'MAHC$^{\triangledown}$'
                    newdata1 = {'Network': network, 'CorRate': corcon, 'Method': alg1,
                               'Metric': 'SHD', 'Value': AvgSHD}
                    newdata2 = {'Network': network, 'CorRate': corcon, 'Method': alg1,
                               'Metric': 'F1', 'Value': AvgF1}
                    newdata3 = {'Network': network, 'CorRate': corcon, 'Method': alg1,
                               'Metric': r'F1$^c$', 'Value': AvgF1_con}
                    plotdata = plotdata.append(newdata1, ignore_index=True)
                    plotdata = plotdata.append(newdata2, ignore_index=True)
                    plotdata = plotdata.append(newdata3, ignore_index=True)

        support(plotdata, index1= 'Network', index2= 'Metric', index3 = 'CorRate')
        plotdata.to_csv(os.getcwd() +  f'/Plots/NumofCorrectConstraints/diarrhoeaandproperty2.csv')

    if mode == 'n_wrocon':
        plotdata = pd.DataFrame(columns=['Network', 'WroRate', 'Method', 'Metric', 'Value'])

        condition1 = (results['NumOfData'] == 100) & (results['CorRate'] == 0.2)
        results = results[condition1]
        # for network in ['ASIA', 'SPORTS', 'DIARRHOEA', 'PROPERTY', 'ALARM', 'FORMED']:
        for network in ['ALARM', 'FORMED']:
            for worcon in [1, 2]:
                for alg in ['UpCM', 'PCstable_con', 'GES_con', 'MAHC_con']:
                    condition2 = (results['Network'] == network) & (results['WroRate'] == worcon) & (results['Method'] == alg) & (results['Repeat'] > -1000000) #delete -inf score
                    temp_results = results[condition2]
                    for i in range(temp_results.shape[0]):

                        SHD = temp_results.iloc[i]['SHD']
                        F1 = temp_results.iloc[i]['F1']
                        F1_con = temp_results.iloc[i]['Con_F1']
                        if alg == 'UpCM':
                            alg1 = 'UpCM'
                        elif alg == 'PCstable_con':
                            alg1 = r'PCstable$^{\triangledown}$'
                        elif alg == 'GES_con':
                            alg1 = r'GES$^{\triangledown}$'
                        elif alg == 'MAHC_con':
                            alg1 = r'MAHC$^{\triangledown}$'
                        newdata1 = {'Network': network, 'WroRate': worcon, 'Method': alg1,
                                   'Metric': 'SHD', 'Value': SHD}
                        newdata2 = {'Network': network, 'WroRate': worcon, 'Method': alg1,
                                   'Metric': 'F1', 'Value': F1}
                        newdata3 = {'Network': network, 'WroRate': worcon, 'Method': alg1,
                                   'Metric': r'F1$^c$', 'Value': F1_con}
                        plotdata = plotdata.append(newdata1, ignore_index=True)
                        plotdata = plotdata.append(newdata2, ignore_index=True)
                        plotdata = plotdata.append(newdata3, ignore_index=True)

        support_box(plotdata, index1= 'Network', index2= 'Metric')
        plotdata.to_csv(os.getcwd() +  f'/Plots/RatioOfwrongCon/diarrhoeaandproperty2.csv')

    if mode == 'n_evaluations':
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 7), sharex= False, sharey= False)
        i = 0
        j = 0
        colors = [(189 / 255, 60 / 255, 51 / 255), (249 / 255, 198 / 255, 118 / 255), (127 / 255, 169 / 255, 205 / 255),
                  (63 / 255, 96 / 255, 163 / 255)]
        for network in ['ASIA', 'SPORTS', 'DIARRHOEA', 'PROPERTY', 'ALARM', 'FORMED']:
            ax = axes[i, j]
            k=0
            for ne in [10,100,500,1000]:
                condition2 = (results['Network'] == network) & (results['parametervalue'] == ne) #delete -inf score
                plotdata = results[condition2]
                plotdata = plotdata.sort_values(by=['score'], ascending=False)

                x_data = plotdata['score']  # 替换为你的横坐标列名
                y_data = plotdata['avgnmi']  # 替换为你的纵坐标列名
                ax.plot(x_data, y_data, label= f'UpCM:iter={ne}', marker='o', linestyle='--', color=colors[k])
                k+=1
            j += 1
            if j > 2:
                j = 0
                i += 1

            ax.set_title(f'Network:{network}', fontsize = 12)
            # ax.set_xlim(ax.get_xlim()[::-1])
            ax.set_xlabel('Score', fontsize = 12)  # 替换为你的横坐标标签
            ax.set_ylabel('Knowledge (NMI)', fontsize = 12)  # 替换为你的纵坐标标签
            # ax.legend()

        handles, labels = axes[0, 0].get_legend_handles_labels()
        labels = ['Maxiter = 10', 'Maxiter = 100', 'Maxiter = 500', 'Maxiter = 1000']
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0), ncol=4)
        plt.tight_layout(rect=[0, 0.03, 1, 1])
        plt.show()

    if mode == 'weights':
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 7), sharex= False, sharey= False)
        i = 0
        j = 0
        colors = [(189 / 255, 60 / 255, 51 / 255), (249 / 255, 198 / 255, 118 / 255), (127 / 255, 169 / 255, 205 / 255),
                  (63 / 255, 96 / 255, 163 / 255)]
        for network in ['ASIA', 'SPORTS', 'DIARRHOEA', 'PROPERTY', 'ALARM', 'FORMED']:
            ax = axes[i, j]
            k=0
            for weight in ['No bias', 'Knowledge bias', 'Data bias']:
                condition2 = (results['Network'] == network) & (results['parametervalue'] == weight) #delete -inf score
                plotdata = results[condition2]
                plotdata = plotdata.sort_values(by=['score'], ascending=False)

                x_data = plotdata['score']  # 替换为你的横坐标列名
                y_data = plotdata['avgnmi']  # 替换为你的纵坐标列名
                ax.plot(x_data, y_data, label= f'UpCM:{weight}', marker='o', linestyle='--', color = colors[k])
                k+=1
            j += 1
            if j > 2:
                j = 0
                i += 1

            ax.set_title(f'Network:{network}', fontsize = 12)
            # ax.set_xlim(ax.get_xlim()[::-1])
            ax.set_xlabel('Score', fontsize = 12)  # 替换为你的横坐标标签
            ax.set_ylabel('Knowledge (NMI)', fontsize = 12)  # 替换为你的纵坐标标签
            #ax.legend()

        handles, labels = axes[0, 0].get_legend_handles_labels()
        labels = ['No bias', 'Knowledge bias', 'Data bias']
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0), ncol=4)
        plt.tight_layout(rect=[0, 0.03, 1, 1])
        plt.show()



def support(results:pd.DataFrame, index1:str, index2:str, index3:str):

    index1s = results[index1].unique()
    index2s = results[index2].unique()

    fig, axes = plt.subplots(nrows=len(index1s), ncols=len(index2s), figsize=(9, 6), sharex=False)

    for i, cond1 in enumerate(index1s):
        for j, cond2 in enumerate(index2s):

            subset = results[(results[index1] == cond1) & (results[index2] == cond2)]

            # 针对当前组合的子图
            ax = axes[i, j]
            metric = subset['Metric'].unique()

            # 遍历每个算法，绘制折线图
            algorithms = subset['Method'].unique()
            colors = [(189/255, 60/255, 51/255), (249/255, 198/255, 118/255), (127/255, 169/255, 205/255), (63/255, 96/255, 163/255)]
            k= 0
            for algorithm in algorithms:
                data = subset[subset['Method'] == algorithm]
                x_data = data[index3]  # 替换为你的横坐标列名
                y_data = data['Value']  # 替换为你的纵坐标列名
                ax.plot(x_data, y_data, label=algorithm, marker='o', color=colors[k])
                k += 1

            # 设置子图标题和标签
            ax.set_title(f'Network:{cond1}', fontsize = 12)
            if i == len(index1s) - 1:
                ax.set_xlabel('Size of dataset', fontsize = 12, family = 'Times New Roman')  # 替换为你的横坐标标签
            ax.set_ylabel(metric[0], fontsize = 12)  # 替换为你的纵坐标标签
            # ax.legend()
            ax.grid(True)

    # 调整子图之间的间距
    # plt.figure(figsize=(3, 1))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0),  ncol=4)
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.show()

def support_box(results:pd.DataFrame, index1:str, index2:str):

    index1s = results[index1].unique()
    index2s = results[index2].unique()

    fig, axes = plt.subplots(nrows=len(index1s), ncols=len(index2s), figsize=(9, 6), sharex=True)
    colors = [(189 / 255, 60 / 255, 51 / 255), (249 / 255, 198 / 255, 118 / 255), (127 / 255, 169 / 255, 205 / 255),
              (63 / 255, 96 / 255, 163 / 255)]
    k = 0
    for i, cond1 in enumerate(index1s):
        for j, cond2 in enumerate(index2s):

            subset = results[(results[index1] == cond1) & (results[index2] == cond2)]

            # 针对当前组合的子图
            ax = axes[i, j]

            # 遍历每个算法，绘制箱线图
            metric = subset['Metric'].unique()

            # subsubset1 = subset[subset['WroRate'] == 1]
            # plot_data = pd.DataFrame(columns=algorithms)
            # for algorithm in algorithms:
            #     temp = subsubset1[subsubset1['Method'] == algorithm]
            #     temp2 = temp['Value'].tolist()
            #     nan_count = 10 - len(temp2)
            #     temp2.extend([np.nan] * nan_count)
            #     plot_data[algorithm] = temp2
            #
            # subsubset2 = subset[subset['WroRate'] == 2]
            # plot_data2 = pd.DataFrame(columns=algorithms)
            # for algorithm in algorithms:
            #     temp = subsubset2[subsubset2['Method'] == algorithm]
            #     temp2 = temp['Value'].tolist()
            #     nan_count = 10 - len(temp2)
            #     temp2.extend([np.nan] * nan_count)
            #     plot_data2[algorithm] = temp2

            # ax.boxplot(plot_data, positions=np.arange(len(algorithms)) * 2 - 0.4, widths=0.6)
            # ax.boxplot(plot_data2, positions=np.arange(len(algorithms)) * 2 + 0.4, widths=0.6)
            sns.boxplot(x='WroRate', y='Value',hue='Method', data=subset,ax=ax,palette=colors)

            ax.tick_params(axis='x', labelsize=12)
            ax.tick_params(axis='y', labelsize=12)
            # 设置标题和坐标轴标签
            ax.set_title(f'Network:{cond1}', fontsize = 12)
            ax.set_xlabel('Incorrect constraint ratio', fontsize = 12)
            ax.set_ylabel(metric[0], fontsize = 12)
            ax.legend().set_visible(False)

    # 调整子图之间的间距
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0),  ncol=4)
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.show()





