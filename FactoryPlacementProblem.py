import random
import cvxpy as cp
import numpy as np
from scipy import linalg as la
from matplotlib import pyplot as plt

'''
Given positions of each cities, select k cities each to construct
a factory, and assign each city to a factory. Such that the total distance 
from the city to the assigned factory is minimized.
pos: position matrix [[x1, y1], [x2, y2]...]
k: number of factories to construct
'''
def k_mean_cluster(pos, k):
    n = pos.shape[0]
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dist = la.norm(pos[i, :] - pos[j, :])
                D[i, j] = dist

    X = cp.Variable((n, n), integer=True)
    constraints = []
    for i in range(n):
        for j in range(n):
            constraints += [X[i, j] >= 0]
            constraints += [X[i, j] <= 1]

    constraints += [cp.sum(cp.norm_inf(X, 0)) <= k]
    constraints += [X@np.ones(n) == np.ones(n)]

    problem = cp.Problem(
        cp.Minimize(cp.sum(cp.multiply(X, D))),
        constraints)
    problem.solve()

    print(problem.status)
    print(X.value)
    return X.value

'''
Given positions of each cities, find minimum k cities each to construct
a factory, and assign each city to a factory. Such that the total distance 
from the city to the assigned factory is minimized. Each city has its own
demand of goods from factory, and each factory has a fixed capacity.
pos: position matrix [[x1, y1], [x2, y2]...]
k: number of factories to construct
demand: demand vector, each entry represent a city. [d1, d2, ...]
factory_capacity: scalar, capacity of a factory.
'''
def k_mean_cluster_with_demand(pos, k, demand, factory_capacity):
    n = pos.shape[0]
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dist = la.norm(pos[i, :] - pos[j, :])
                D[i, j] = dist

    X = cp.Variable((n, n), integer=True)
    constraints = []
    for i in range(n):
        for j in range(n):
            constraints += [X[i, j] >= 0]
            constraints += [X[i, j] <= 1]

    constraints += [cp.sum(X) == n]
    constraints += [cp.sum(cp.norm_inf(X, 0)) <= k]
    constraints += [X@np.ones(n) == np.ones(n)]
    constraints += [X.T@demand <= np.ones(n) * factory_capacity]

    problem = cp.Problem(
        cp.Minimize(cp.sum(cp.multiply(X, D))),
        constraints)
    problem.solve()

    print(problem.status)
    print(X.value)
    print(f"demand: {demand}")
    return X.value


if __name__=='__main__':
    random.seed(100)
    random_gen_center = np.array([
        [0, 0],
        [10, 10],
        [10, 20],
        [5, 8]
    ])
    cities = []
    demands = []

    radius = 5
    sample_per_cluster = 10
    k_cluster = 3
    demands_max = 5
    factory_capacity = 15

    # Randomly generate cities
    for i in range(random_gen_center.shape[0]):
        center = random_gen_center[i, :]
        for s in range(sample_per_cluster):
            x = random.uniform(-radius, radius)
            y = random.uniform(-radius, radius)
            cities.append(center + np.array([x, y]))
            demands.append(random.uniform(0, demands_max))

    pos = np.array(cities)
    demands = np.array(demands)

    cluster_mat = k_mean_cluster(pos, k_cluster)
    # cluster_mat = k_mean_cluster_with_demand(
    #     pos,
    #     np.ceil(np.sum(demands)/factory_capacity),
    #     demands,
    #     factory_capacity)

    clusters = []
    clusters_centers = []
    for col in range(cluster_mat.shape[0]):
        cluster = []
        for row in range(cluster_mat.shape[0]):
            if cluster_mat[row, col] != 0:
                cluster.append(pos[row, :])

        if len(cluster) != 0:
            clusters.append(np.array(cluster))
            clusters_centers.append(pos[col, :])

    plt.figure()
    plt.xlim([-10, 30])
    plt.ylim([-10, 30])
    plt.scatter(pos[:, 0], pos[:, 1])

    plt.figure()
    plt.xlim([-10, 30])
    plt.ylim([-10, 30])
    colors=['b', 'y', 'g', 'r', 'black', 'skyblue']
    for i in range(len(clusters)):
        X = clusters[i][:, 0]
        Y = clusters[i][:, 1]
        c_idx = i%len(colors)
        plt.scatter(X, Y, c=colors[c_idx], marker='x')
        plt.scatter(clusters_centers[i][0], clusters_centers[i][1], c=colors[c_idx])

    plt.show()