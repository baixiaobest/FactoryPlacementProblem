import cvxpy as cp
import numpy as np
from scipy import linalg as la
from matplotlib import pyplot as plt
from geopy.distance import geodesic
import pickle
import numpy as np
from matplotlib import font_manager

'''
lat_long: numpy nx2, [[latitude, longitude], ...] matrix of city locations.
population: Population of each city in numpy 1xn, unit in 1e4 people.
factory_capacity: Population limit each factory can serve, unit in 1e4 people.
k: Number of factory to build.
d_max: Maximum reach of each factory.
'''
def facotory_placement(lat_long, population, k, factory_capacity=np.inf, d_max=np.inf):
    n = lat_long.shape[0]
    # Distance matrix
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dist = geodesic(lat_long[i, :], lat_long[j, :]).km
                D[i, j] = dist

    # Total population
    p_sum = np.sum(population)
    # Weight Vector, weight is determined by the population of the cities.
    w_vec = population / p_sum

    # Cost matrix, Cij represents the cost of shipping goods from factory j to city i per unit distance.
    # Cij is the product of distance from factory j to city i and the population weight of city i.
    C = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                C[i, j] = w_vec[i] * D[i, j]

    X = cp.Variable((n, n), integer=True)
    constraints = []

    # Add maximum distance constraint and binary value constraint on X.
    for i in range(n):
        for j in range(n):
            if D[i, j] > d_max:
                # maximum distance constraint.
                constraints += [X[i, j] == 0]
            else:
                # binary value constraint.
                constraints += [X[i, j] >= 0]
                constraints += [X[i, j] <= 1]

    # Factory number constraint. Total K factory needs to be built.
    constraints += [cp.sum(cp.norm_inf(X, 0)) <= k]
    # Factory capacity constraint.
    if factory_capacity != np.inf:
        constraints += [X.T@population <= factory_capacity * np.ones(n)]
    # Each city needs to be assigned a factory.
    constraints += [X@np.ones(n) == np.ones(n)]

    problem = cp.Problem(
        cp.Minimize(cp.sum(cp.multiply(X, C))),
        constraints)
    problem.solve()

    print(problem.status)
    return X.value, C

def plot_cities(utm_list, city_names):
    # Plot cities before assignments
    plt.figure()
    plt.scatter(utm_list[:, 0], utm_list[:, 1])
    plt.xlim([-1.5e6, 1.5e6])
    plt.ylim([2e6, 5e6])

    for i in range(utm_list.shape[0]):
        name = city_names[i]
        ax = plt.gca()
        ax.annotate(name, (utm_list[i, 0], utm_list[i, 1]))

def plot_city_clusters(clusters, clusters_centers, city_names, utm_list):
    # Plot cities after assignment.
    plt.figure()
    plt.xlim([-2e6, 1.5e6])
    plt.ylim([2e6, 5.5e6])
    colors = ['b', 'y', 'g', 'r', 'black', 'skyblue',
              'olive', 'aqua', 'maroon', 'lightgreen', 'teal', 'blueviolet']
    for i in range(len(clusters)):
        X = clusters[i][:, 0]
        Y = clusters[i][:, 1]
        c_idx = i % len(colors)
        plt.scatter(X, Y, c=colors[c_idx], marker='x')
        plt.scatter(clusters_centers[i][0], clusters_centers[i][1], c=colors[c_idx])

    for i in range(utm_list.shape[0]):
        name = city_names[i]
        ax = plt.gca()
        ax.annotate(name, (utm_list[i, 0], utm_list[i, 1]))

def plot_cluster_info(cluster_names, cluster_sizes, cluster_population, cluster_cost):
    plt.figure()
    ax = plt.subplot(2, 1, 1)
    ax.bar(range(len(cluster_sizes)), cluster_sizes, tick_label=cluster_names)
    ax.set_title("细胞中心位置vs供应城市数量")
    ax.set_ylabel("供应城市数量")

    ax = plt.subplot(2, 1, 2)
    ax.bar(range(len(cluster_population)), cluster_population, tick_label=cluster_names)
    ax.set_title("细胞中心位置vs供应人口")
    ax.set_ylabel("供应人口(万人)")

    plt.figure()
    plt.gca().bar(range(len(cluster_cost)), cluster_cost, tick_label=cluster_names)
    plt.gca().set_title("细胞中心位置vs相对运输支出")
    plt.gca().set_ylabel("相对运输支出(%)")

if __name__=="__main__":
    geo_datafile = open('data/geodata', 'rb')
    geo_data = pickle.load(geo_datafile)
    geo_datafile.close()

    pop_datafile = open('data/populationdata', 'rb')
    pop_data = pickle.load(pop_datafile)
    pop_datafile.close()

    lat_long = geo_data['lat_long']
    utm_list = geo_data['utm_list']
    city_names = geo_data['cities']
    city_names_en = geo_data['city_names_en']

    # Construct population vector.
    population_vec = np.zeros(len(city_names))
    for i in range(len(city_names)):
        name = city_names[i]
        population_vec[i] = pop_data[name]

    cluster_mat, cost_mat = facotory_placement(lat_long, population_vec, k=13, factory_capacity=5000, d_max=1000)

    # Process cluster matrix
    clusters = [] # List of clusters, each cluster is numpy (ix2) containing positions.
    cluster_total_population = [] # List of cluster total population.
    clusters_centers = [] # List of numpy size 2 vector containing cluster center position.
    cluster_center_names = [] # Name of the cluster center.
    cluster_cost = [] # List of cluster cost. Percentage cost. Percentage of total clusters operating cost.
    for col in range(cluster_mat.shape[0]):
        cluster = []
        cost = 0
        cluster_population = 0
        for row in range(cluster_mat.shape[0]):
            if cluster_mat[row, col] != 0:
                cluster.append(utm_list[row, :])
                cost += cost_mat[row, col]
                cluster_population += population_vec[row]

        if len(cluster) != 0:
            clusters.append(np.array(cluster))
            clusters_centers.append(utm_list[col, :])
            cluster_center_names.append(city_names[col])
            cluster_total_population.append(cluster_population)
            cluster_cost.append(cost)

    cluster_cost = cluster_cost / sum(cluster_cost) * 100
    cluster_sizes = [c.shape[0] for c in clusters]

    print(cluster_center_names)

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rc('axes', unicode_minus=False)

    plot_cities(utm_list, city_names)
    plot_city_clusters(clusters, clusters_centers, city_names, utm_list)
    plot_cluster_info(cluster_center_names, cluster_sizes, cluster_total_population, cluster_cost)

    plt.show()