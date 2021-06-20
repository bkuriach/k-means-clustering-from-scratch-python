import scipy.io
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

Numpyfile= scipy.io.loadmat("C:\\Projects\\MCS\\CSE575\\Project 2 - KMeans\\data.mat")
data = pd.DataFrame(Numpyfile['AllSamples'])
data.columns=['x1','x2']

m = data.shape[0]
n = data.shape[1]

n_iter = 100
K = 5

def assign(data,Centroids):

    EuclDistance = np.array([]).reshape(m, 0)
    for k in range(K):
        tempDist = np.sum((data - Centroids[:, k]) ** 2, axis=1)
        EuclDistance = np.c_[EuclDistance, tempDist]
    Clusters = np.argmin(EuclDistance, axis=1) + 1

    return(Clusters)


def map_cluster_data(data, K):
    clusterDataMap = {}
    for k in range(K):
        clusterDataMap[k + 1] = np.array([]).reshape(2, 0)

    for i in range(m):
        clusterDataMap[clusters[i]] = np.c_[clusterDataMap[clusters[i]], data.iloc[i]]

    for k in range(K):
        clusterDataMap[k + 1] = clusterDataMap[k + 1].T

    return(clusterDataMap)

def centroid(clusterDataMap,Centroids):

    for k in range(K):
        Centroids[:, k] = np.mean(clusterDataMap[k + 1], axis=0)

    return(Centroids)

def initialize_centroids(data, K):
    Centroids = np.array([]).reshape(data.shape[1], 0)
    for i in range(K):
        rand = random.randint(0, data.shape[0] - 1)
        Centroids = np.c_[Centroids, data.iloc[rand]]
    return(Centroids)


K=3
Centroids = initialize_centroids(data,K)
for i in range(n_iter):
    clusters = assign(data, Centroids)
    clusterDataMap = map_cluster_data(data, K)
    Centroids = centroid(clusterDataMap, Centroids)

plt.scatter(data.iloc[:,0],data.iloc[:,1],c='black',label='Unclustered data')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.title('Plot of data points')
plt.show()

color=['red','blue','green','cyan','magenta','grey', 'yellow', 'orange', 'black', 'purple']
labels=['cluster1','cluster2','cluster3','cluster4','cluster5','cluster6', 'cluster7','cluster8','cluster9','cluster10']
for k in range(K):
    plt.scatter(clusterDataMap[k+1][:,0],clusterDataMap[k+1][:,1],c=color[k],label=labels[k])
plt.scatter(Centroids[0,:],Centroids[1,:],s=300,c='yellow',label='Centroids')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.show()

WCSS_array=np.array([])
for K in range(2,11):
    print(K)
    Centroids = initialize_centroids(data, K)
    for i in range(n_iter):
        clusters = assign(data, Centroids)
        clusterDataMap = map_cluster_data(data, K)
        Centroids = centroid(clusterDataMap, Centroids)
    wcss = 0
    for k in range(K):
        wcss += np.sum((clusterDataMap[k + 1] - Centroids[:, k]) ** 2)
    WCSS_array = np.append(WCSS_array, wcss)

K_array=np.arange(2,11,1)
plt.plot(K_array,WCSS_array)
plt.xlabel('Number of Clusters')
plt.ylabel('within-cluster sums of squares (WCSS)')
plt.title('Elbow method to determine optimum number of clusters')
plt.show()


def initialize_centroids_strategy2(data, K):
    Centroids = np.array([]).reshape(data.shape[1], 0)
    # centroidIndexes = []
    for i in range(K):
        if i ==0:
            rand = random.randint(0, data.shape[0] - 1)
            Centroids = np.c_[Centroids, data.iloc[rand]]
            # centroidIndexes.append(rand)
            data=data.drop(data.index[rand])
        else:
            centroidMean = np.mean(Centroids,axis=1)
            index = np.argmax(np.sqrt(np.sum((data - centroidMean) ** 2, axis=1)), axis=1)
            # centroidIndexes.append(index)
            Centroids = np.c_[Centroids, data.iloc[index]]
            data = data.drop(data.index[index])
    return(Centroids)

WCSS_array=np.array([])
for K in range(2,11):
    print(K)
    for i in range(n_iter):
        Centroids = initialize_centroids_strategy2(data, K)
        clusters = assign(data, Centroids)
        clusterDataMap = map_cluster_data(data, K)
        Centroids = centroid(clusterDataMap, Centroids)
        wcss = 0
    for k in range(K):
        wcss += np.sum((clusterDataMap[k + 1] - Centroids[:, k]) ** 2)
    WCSS_array = np.append(WCSS_array, wcss)

K_array=np.arange(2,11,1)
plt.plot(K_array,WCSS_array)
plt.xlabel('Number of Clusters')
plt.ylabel('within-cluster sums of squares (WCSS)')
plt.title('Elbow method to determine optimum number of clusters')
plt.show()

#
#
# WCSS_array=np.array([])
# for K in range(2,11):
#     Centroids = initialize_centroids(data, K)
#     for i in range(n_iter):
#         clusters = assign(data, Centroids)
#         clusterDataMap = map_cluster_data(data, K)
#         Centroids = centroid(clusterDataMap, Centroids)
#     wcss = 0
#     # Compute Objective functions
#     for k in range(K):
#         wcss += np.sum((clusterDataMap[k + 1] - Centroids[:, k]) ** 2)
#     WCSS_array = np.append(WCSS_array, wcss)
#
# # Plot the objective function
# KMeans_array=np.arange(2,11,1)
# plt.figure()
# plt.plot(KMeans_array,WCSS_array)
# plt.xlabel('Number of Clusters')
# plt.ylabel('within-cluster sums of squares (WCSS)')
# plt.title('Strategy1 - Run 2: Elbow Chart to identify optimum cluster number')
# plt.show()
