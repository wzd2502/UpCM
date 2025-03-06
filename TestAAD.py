import numpy as np

import sys

import unittest
from pickle import load

import numpy as np
import pandas as pd

from causallearn.search.FCMBased import lingam



def test_DirectLiNGAM():
    np.set_printoptions(precision=3, suppress=True)
    np.random.seed(100)
    x3 = np.random.uniform(size=1000)
    x0 = 3.0 * x3 + np.random.uniform(size=1000)
    x2 = 6.0 * x3 + np.random.uniform(size=1000)
    x1 = 3.0 * x0 + 2.0 * x2 + np.random.uniform(size=1000)
    x5 = 4.0 * x0 + np.random.uniform(size=1000)
    x4 = 8.0 * x0 - 1.0 * x2 + np.random.uniform(size=1000)
    X = pd.DataFrame(np.array([x0, x1, x2, x3, x4, x5]).T, columns=['x0', 'x1', 'x2', 'x3', 'x4', 'x5'])
    truedag = [[0,0,0,3,0,0],
               [3,0,2,0,0,0],
               [0,0,0,6,0,0],
               [0,0,0,0,0,0],
               [8,0,-1,0,0,0],
               [4,0,0,0,0,0]]

    model = lingam.DirectLiNGAM()
    model.fit(X)

    print(model.causal_order_)
    print(model.adjacency_matrix_)
    linresu = model.adjacency_matrix_
    return X, truedag, linresu


import numpy as np
from scipy.optimize import minimize


def objective_function(W, X):
    n = W.shape[0]
    term1 = 0.5 / n * np.linalg.norm(X - X @ W, 'fro') ** 2
    term2 = 0.5 * np.sum(np.abs(W))
    return term1 + term2


def constraint(W):
    return np.trace(np.exp(W * W)) - W.shape[0]


def optimize(X):
    n = X.shape[1]
    initial_guess = np.random.randn(n, n)  # Initial guess for W

    def objective(W):
        return objective_function(W.reshape((n, n)), X)

    def constraint_eq(W):
        return constraint(W.reshape((n, n)))

    # Define the optimization problem
    cons = {'type': 'eq', 'fun': constraint_eq}
    bounds = [(None, None)] * (n * n)  # No bounds for W

    # Minimize the objective function subject to the constraint
    result = minimize(objective, initial_guess.flatten(), method='SLSQP', constraints=cons, bounds=bounds)

    # Reshape the optimized W to n x n matrix
    W_opt = result.x.reshape((n, n))

    return W_opt

X, truedag, lingresu = test_DirectLiNGAM()
llmresu = optimize(X.to_numpy())
print(llmresu)

mse1= np.mean((truedag - llmresu) ** 2)
mse2= np.mean((truedag - lingresu) ** 2)
print(mse1, mse2)


