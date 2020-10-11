#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon 20th Apr 2020

@author: christoph
"""

import numpy as np
from tqdm import tqdm
import mom
import pickle
import pandas as pd
import seaborn as sns
import networkx as nx
from sklearn.metrics import mean_squared_error
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

np.set_printoptions(linewidth=200)
np.set_printoptions(suppress=True)


def align_columns(A, B):
    k = A.shape[1]
    G = nx.Graph()
    G.add_nodes_from(range(0, k), bipartite=0)
    G.add_nodes_from(range(k, k*2), bipartite=1)
    elist = []
    for i in range(0, k):
        for j in range(0, k):
            # elist.append((i,k+j,wasserstein_distance(L2[h,:,i],L2R3[h,:,j])))
            elist.append((i, k+j, np.abs(A[:, i]-B[:, j]).sum()))
    maximum = max(elist, key=lambda t: t[2])[2]
    elist = list(map(lambda x: (x[0], x[1], maximum-x[2]), elist))
    G.add_weighted_edges_from(elist)
    matching = nx.max_weight_matching(G)
    sorted_matching = sorted([sorted(pair) for pair in matching])
    indexes = np.array(list(map(lambda x: x[1]-k, sorted_matching)))
    return indexes


def gen_data(k, c, d, l, alpha0, n):
    m = c+d
    # Draw alpha
    alpha = np.random.randint(1, high=5, size=k)
    alpha = alpha / alpha.sum() * alpha0

    # Draw theta
    beta = np.array([1/d] * d)
    theta = np.zeros((k, c+d))
    for i in range(k):
        # Draw continous parameters
        theta[i, :c] = np.random.randint(-10, high=10, size=c)
        # Draw discrete parameters
        theta[i, c:] = np.random.dirichlet(beta, 1)

    # Draw  sigma
    sigma = np.random.randint(1, high=5, size=k)

    # Draw samples
    x = np.zeros((n, c+d))
    # Sample distribution of topics
    omegas = np.random.dirichlet(alpha, n)
    for m in tqdm(range(n)):
        # Skip randoom document length for easier data handling
        # n_w = np.random.poisson(xi)
        n_w = l
        # Sample how often each topic occurs in the current document
        z = np.random.multinomial(l, omegas[m])
        for t in range(k):
            # Sample discrete variables for topic t
            x[m, c:] += np.random.multinomial(z[t], theta[t, c:])

    # Sample all continous variables at once for speed up
    x[:, :c] = 0
    for t in range(k):
        x[:, :c] = x[:, :c] + np.multiply(np.random.multivariate_normal(
            theta[t, :c],  sigma[t]*np.identity(c), n), (omegas[:, t])[:, np.newaxis])

    return [alpha, theta, sigma, x]


ks = [5, 10, 20]
c = 50
d = 100
l = 30
alpha0 = 1
ns = [1000, 2500, 5000, 10000, 25000, 50000]

alpha_errors = {}
theta_errors = {}
# init error lists
for k in ks:
    alpha_errors[k] = {'mse': [], 'norm': []}
    theta_errors[k] = {'mse': [], 'norm': []}

np.random.seed(27)
# First one seems to be trash
alpha, theta, sigma, x = gen_data(10, c, d, l, alpha0, 1000)

for k in ks:
    for n in ns:
        for i in range(5):
            alpha_errors[k]['mse'].append([])
            alpha_errors[k]['norm'].append([])
            theta_errors[k]['mse'].append([])
            theta_errors[k]['norm'].append([])
            alpha, theta, sigma, x = gen_data(k, c, d, l, alpha0, n)
            mu, ealpha, *_ = mom.fit(x, c, alpha0, k)
            toppic_mapping = align_columns(theta.T, mu)
            ealpha = ealpha[toppic_mapping]
            mu = mu[:, toppic_mapping]
            alpha_errors[k]['mse'][i].append(mean_squared_error(alpha, ealpha))
            alpha_errors[k]['norm'][i].append(np.linalg.norm(alpha-ealpha,))
            theta_errors[k]['mse'][i].append(mean_squared_error(theta.T, mu))
            theta_errors[k]['norm'][i].append(np.linalg.norm(theta.T-mu, 'fro'))


# pickle.dump(alpha_errors, open('alpha_errors4.pkl', 'wb'))
# pickle.dump(theta_errors, open('theta_errors4.pkl', 'wb'))


# alpha_errors = pickle.load(open('alpha_errors4.pkl', 'rb'))
# theta_errors = pickle.load(open('alpha_errors4.pkl', 'rb'))

df_data = {'n': [], 'norm': [], 'k': []}
j = 0
for n in ns:
    for i in range(5):
        for k in [5, 20]:
            df_data['n'].append(n)
            df_data['k'].append(k)
            df_data['norm'].append(theta_errors[k]['norm'][i][j])
    j = j+1
df = pd.DataFrame(data=df_data)

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})
g = sns.relplot(x="n", y="norm", hue="k", style="k", palette={5: sns.color_palette('tab10')
                                                              [0], 20: sns.color_palette('tab10')[1]
                                                              },
                dashes=True, height=3.5, aspect=1.4, markers=True, kind="line", data=df, legend='full')
g.set(xlim=(min(ns), max(ns)))
g.savefig('synthetic.pgf', bbox_inches='tight', pad_inches=0)
