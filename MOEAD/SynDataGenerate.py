import networkx as nx
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.sampling import BayesianModelSampling
import numpy as np
import os
import pandas as pd
# Function to assign random CPDs to a Bayesian Network
def assign_random_cpds(model):
    for node in model.nodes():
        parents = list(model.predecessors(node))
        if not parents:
            # Marginal distribution for nodes without parents
            cpd = TabularCPD(
                variable=node,
                variable_card=2,  # Binary variable
                values=np.random.dirichlet(np.ones(2), 1).T,  # Random probabilities
            )
        else:
            # Conditional distribution for nodes with parents
            parent_card = [2] * len(parents)  # Binary parents
            cpd = TabularCPD(
                variable=node,
                variable_card=2,  # Binary variable
                values=np.random.dirichlet(np.ones(2), 1 << len(parents)).T,  # Random probabilities
                evidence=parents,
                evidence_card=parent_card,
            )
        model.add_cpds(cpd)
    return model

# Function to generate a random DAG using Erdős-Rényi model
def generate_er_dag(n, p, model_name, num_samples):
    dag = nx.erdos_renyi_graph(n, p, directed=False)
    adj_matrix = nx.adjacency_matrix(dag).toarray()
    adj_matrix = np.triu(adj_matrix, k = 1)
    row_indices, col_indices = np.nonzero(adj_matrix)

    SBN = BayesianNetwork()
    SBN.add_nodes_from(nodes = [f'X{i}' for i in range(n)])
    SBN.add_edges_from([(f'X{row}', f'X{col}') for row, col in zip(row_indices, col_indices)])
    SBN = assign_random_cpds(SBN)

    edge_data = []

    # 遍历贝叶斯网络的每一条边
    for i, (parent, child) in enumerate(SBN.edges()):
        edge_data.append({
            'ID': i + 1,  # ID从1开始
            'Variable 1': parent,  # 父节点
            'Dependency': '->',  # Dependency列是'->'
            'Variable 2': child   # 子节点
        })

    # 将数据转换为DataFrame
    df = pd.DataFrame(edge_data)
    df.to_csv(f"Networks\\{model_name}.csv",index=False)

    sampler = BayesianModelSampling(SBN)
    samples = sampler.forward_sample(size=num_samples)
    samples.to_csv(f'Datasets\\{model_name}.csv', index=False)

    return SBN

# Function to generate a random DAG using Scale-Free model
def generate_sf_dag(n, m, model_name,num_samples):
    dag = nx.barabasi_albert_graph(n, m)

    adj_matrix = nx.adjacency_matrix(dag).toarray()
    adj_matrix = np.triu(adj_matrix, k = 1)
    row_indices, col_indices = np.nonzero(adj_matrix)

    SBN = BayesianNetwork()
    SBN.add_nodes_from(nodes = [f'X{i}' for i in range(n)])
    SBN.add_edges_from([(f'X{row}', f'X{col}') for row, col in zip(row_indices, col_indices)])
    SBN = assign_random_cpds(SBN)

    edge_data = []

    # 遍历贝叶斯网络的每一条边
    for i, (parent, child) in enumerate(SBN.edges()):
        edge_data.append({
            'ID': i + 1,  # ID从1开始
            'Variable 1': parent,  # 父节点
            'Dependency': '->',  # Dependency列是'->'
            'Variable 2': child   # 子节点
        })

    # 将数据转换为DataFrame
    df = pd.DataFrame(edge_data)
    df.to_csv(f"Networks\\{model_name}.csv",index=False)

    sampler = BayesianModelSampling(SBN)
    samples = sampler.forward_sample(size=num_samples)
    samples.to_csv(f'Datasets\\{model_name}.csv', index=False)

    return SBN


# Main function to generate DAGs, assign CPDs, and generate samples
def generate():
    # Parameters
    node_sizes = [100, 200, 500]
    er_prob = 0.03  # Probability for ER DAG
    sf_edges = 2   # Number of edges for SF DAG
    num_samples = 10000

    # Create directories to save results

    # Generate ER and SF DAGs
    for n in node_sizes:
        # Generate ER DAG
        model_name = f'Large-ER{n}'
        er_model = generate_er_dag(n, er_prob,model_name, num_samples)


        # Generate SF DAG
        model_name = f'Large-SF{n}'
        sf_model = generate_sf_dag(n, sf_edges, model_name, num_samples)

        print(f"Completed processing for {n} nodes.")


generate()
