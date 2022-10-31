import cvxpy as cp
import numpy as np
from scipy import linalg as la
from matplotlib import pyplot as plt
from geopy.distance import geodesic
import pickle
import numpy as np

'''
lat_long: numpy nx2, [[latitude, longitude], ...] matrix of city locations.
population: Population of each city in numpy 1xn.
k: Number of factory to build.
d_max: Maximum reach of each factory.
'''
def facotory_placement(lat_long, population, k, d_max=np.inf):
    n = lat_long.shape[0]
    # Distance matrix
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dist = geodesic(lat_long[i, :], lat_long[j, :]).km
                D[i, j] = dist

    X = cp.Variable((n, n), integer=True)
    constraints = []
    for i in range(n):
        for j in range(n):
            if D[i, j] > d_max:
                # maximum distance constraint.
                constraints += [X[i, j] == 0]
            else:
                # binary value constraint.
                constraints += [X[i, j] >= 0]
                constraints += [X[i, j] <= 1]

    constraints += [cp.sum(cp.norm_inf(X, 0)) <= k]
    constraints += [X@np.ones(n) == np.ones(n)]

    problem = cp.Problem(
        cp.Minimize(cp.sum(cp.multiply(X, D))),
        constraints)
    problem.solve()

    print(problem.status)
    return X.value

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
    population = np.zeros(len(city_names))
    for i in range(len(city_names)):
        name = city_names[i]
        population[i] = pop_data[name]
        print(f"{name}: {pop_data[name]}")

    cluster_mat = facotory_placement(lat_long, k=10, d_max=500)

    clusters = []
    clusters_centers = []
    cluster_center_names = []
    for col in range(cluster_mat.shape[0]):
        cluster = []
        for row in range(cluster_mat.shape[0]):
            if cluster_mat[row, col] != 0:
                cluster.append(utm_list[row, :])

        if len(cluster) != 0:
            clusters.append(np.array(cluster))
            clusters_centers.append(utm_list[col, :])
            cluster_center_names.append(city_names[col])

    print(cluster_center_names)

    plt.figure()
    plt.scatter(utm_list[:, 0], utm_list[:, 1])
    plt.xlim([-1.5e6, 1.5e6])
    plt.ylim([2e6, 5e6])

    for i in range(utm_list.shape[0]):
        name = city_names_en[i]
        ax = plt.gca()
        ax.annotate(name, (utm_list[i, 0], utm_list[i, 1]))

    plt.figure()
    plt.xlim([-1.5e6, 1.5e6])
    plt.ylim([2e6, 5e6])
    colors = ['b', 'y', 'g', 'r', 'black', 'skyblue']
    for i in range(len(clusters)):
        X = clusters[i][:, 0]
        Y = clusters[i][:, 1]
        c_idx = i % len(colors)
        plt.scatter(X, Y, c=colors[c_idx], marker='x')
        plt.scatter(clusters_centers[i][0], clusters_centers[i][1], c=colors[c_idx])

    for i in range(utm_list.shape[0]):
        name = city_names_en[i]
        ax = plt.gca()
        ax.annotate(name, (utm_list[i, 0], utm_list[i, 1]))

    plt.show()