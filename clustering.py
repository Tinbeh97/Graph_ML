import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from graphfeatures import Degree
import scipy.linalg as la
from collections import Counter
from sklearn.cluster import SpectralClustering, AffinityPropagation, AgglomerativeClustering
from sklearn.cluster import KMeans
from copy import deepcopy
import random
import os
from torch_cluster import graclus_cluster
import torch_geometric.utils as g_utils
from scipy import sparse

#loading matrices
#Dict = '/Users/tina/Documents/EEG_graph_project/simulation/data'
#train_matrix = np.load(Dict+'/cov.npy')

def A_binarize(A_matrix,percent=0.25,Model='cov',sparse=True):
    #threshold
    if(A_matrix.ndim==3):
        n_subject = A_matrix.shape[0]
        n_nodes = A_matrix.shape[1]
    else:
        n_subject = 1
        n_nodes = A_matrix.shape[0]
    A_matrix = A_matrix.reshape(n_subject,n_nodes*n_nodes)
    if Model == 'pli':
        quant = 1-percent
    else:
        quant = percent
    thresh = np.quantile(A_matrix, quant, axis = 1, keepdims=1)
    #binary matrix
    if Model == 'pli':
        bA_matrix = (A_matrix < thresh)
    else:      
        bA_matrix = (A_matrix >= thresh)
    if(not(sparse)):
        if(n_subject==1):
            return bA_matrix.reshape(n_nodes,n_nodes)
        return bA_matrix.reshape(-1,n_nodes,n_nodes)
    A = deepcopy(A_matrix)
    A[~bA_matrix] = 0
    if(n_subject==1):
        return A.reshape(n_nodes,n_nodes)
    return A.reshape(n_subject,n_nodes,n_nodes)

def graph_representation(train_A,graph_num=None,Prop='cluster_C',plotting=True,sort=True,laplacian=False):
    if(graph_num==None):
        graph_num = random.randint(1,len(train_A))-1
    n = train_A.shape[1]
    if(Prop=='degree_D_first'):
        #degree distribution
        train_bA = A_binarize(train_A)
        if(plotting): plt.figure(); u = plt.hist(np.diag(Degree(train_bA[graph_num])))
        m = (u[1][1:len(u[1])] - u[1][0:len(u[1])-1])/2 + u[1][0:len(u[1])-1]
        if(plotting): plt.bar(m,np.divide(u[0],m)) #p(K) = N_K/K
        return m,u
    elif(Prop=='degee_D'):
        m = Counter(np.sort(np.diag(Degree(train_bA[graph_num]))))
        u = np.divide(np.array(list(m.values())),np.array(list(m.keys())))
        if(plotting): plt.bar(m.keys(), u); plt.show()
        return m,u
    elif(Prop=='cluster_C' or Prop=='cluster_C_avg'):
        #clustering coefficient
        A = nx.Graph(train_bA[graph_num])
        if(Prop=='cluster_C_avg'):
            return nx.average_clustering(A)
        c = nx.clustering(A) #np.max(list(c.values()))
        return c
    elif(Prop=='Laplacian'):
        #Laplacian matrix5
        train_L = Degree(train_A)-train_A
        D_inv = Degree((np.sum(train_A,axis=2)**(-0.5)).reshape(train_A.shape[0],train_A.shape[1],1))
        train_Lhat = D_inv * train_L * D_inv
        return train_Lhat
    elif(Prop=='Spectral'):
        #spectral 
        if(laplacian):
            A = Degree(train_A)-train_A
        else:
            A = train_A
        eigvals, eigvecs = la.eig(A[graph_num])
        eigvals = eigvals.real #symmetric
        if(plotting): plt.plot(np.arange(64), np.sort(eigvals),'bo')#number of clusters
        #u = eigvecs.T @ np.diag(eigvals) @ eigvecs
        #np.allclose(A[95],u) #true
        if(sort):
            #sort based on the eigenvalues
            vecs = eigvecs[:,np.argsort(eigvals)]
            vals = eigvals[np.argsort(eigvals)]
            return vals, vecs
        return eigvals, eigvecs
    elif(Prop=='shortest_path_binary'):
        #shortest path for binary A
        G = nx.Graph(train_A[graph_num].reshape(n,n))
        path = nx.shortest_path(G) #binary #max=4
        return path
    elif(Prop=='shortest_path_weighted'):
        #shortest path
        G = nx.Graph(train_A[graph_num].reshape(n,n))
        path = nx.all_pairs_dijkstra_path(G) #weighted
        return path
    elif(Prop=='diameter'):
        #diameter of binary connected graph A
        G = nx.Graph(train_A[graph_num].reshape(n,n))
        return nx.diameter(G)
    elif(Prop=='B_centrality'):
        G = nx.Graph(train_A[graph_num].reshape(n,n))
        return nx.betweenness_centrality(G)
    elif(Prop=='D_centrality'):
        G = nx.Graph(train_A[graph_num].reshape(n,n))
        return nx.degree_centrality(G)
    else:
        raise Exception("non-existing attribute")

def graph_clustering(A_matrix,method,n_clusters,ratio=None,graph_num=None,plotting=True,Mean=False):
    if(graph_num==None):
        graph_num = random.randint(1,len(A_matrix))-1
    if(Mean):
        graph_num = 0; A_matrix = np.mean(A_matrix,axis=0,keepdims=True)
    n = A_matrix.shape[1]
    if(method=='kmeans'):
        #kmeans on first n vectors with nonzero eigenvalues
        _, vecs = graph_representation(train_A=A_matrix,graph_num=graph_num,Prop='Spectral',plotting=False)
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(vecs[:,1:n_clusters].reshape(-1,n_clusters-1))
        if(ratio==None):
            return kmeans.labels_
        num = np.sum(kmeans.labels_)
        ind = 0 if num>(n//2) else 1
        prob = (kmeans.fit_transform(vecs[:,1:n_clusters].reshape(-1,n_clusters-1)))
        thresh = np.quantile(prob[:,ind], ratio)
        return (prob[:,ind] >= thresh)
    elif(method=='Spectral_clustering'):
        adjacency_matrix = A_matrix[graph_num].reshape(n,n)
        sc = SpectralClustering(n_clusters, affinity='precomputed', n_init=100,
                                 assign_labels='discretize')
        Class = sc.fit_predict(adjacency_matrix)
        if(plotting):
            Ab_matrix = A_binarize(A_matrix)
            G = nx.Graph(Ab_matrix[graph_num])
            plt.figure(); nx.draw(G, node_size=200, pos=nx.spring_layout(G)); plt.show()
            plt.figure(); nx.draw(G, node_color=Class, node_size=200, pos=nx.spring_layout(G)); plt.show()
        return Class
    elif(method=='Affinity_propagation'):
        _, vecs = graph_representation(train_A=A_matrix,graph_num=graph_num,Prop='Spectral',plotting=False)
        clustering = AffinityPropagation().fit(vecs[:,1:n_clusters])
    elif(method=='Agglomerative_clustering'):
        _, vecs = graph_representation(train_A=A_matrix,graph_num=graph_num,Prop='Spectral',plotting=False)
        clustering = AgglomerativeClustering(n_clusters=n_clusters).fit(vecs[:,1:n_clusters].reshape(-1,n_clusters-1))
    elif(method=='Graclus'):
        sA = sparse.csr_matrix(A_matrix[graph_num])
        edge_index, edge_weight = g_utils.from_scipy_sparse_matrix(sA)
        cluster = graclus_cluster(edge_index[0], edge_index[1], edge_weight)
        return cluster.numpy()
    else:
        raise Exception("non-existing clustering method")
    return clustering.labels_

def MST(train_A,graph_num=100,printing=False,method=2):
    #MST minimum spanning tree 
    if(graph_num==None):
        graph_num = random.randint(1,len(train_A))-1
    n = train_A.shape[1]
    if(method==1):
        Tcsr = minimum_spanning_tree(train_A[graph_num].reshape(n,n))
        E = Tcsr.toarray().astype(int)
        G3 = nx.Graph(Tcsr)
        return E, G3
    G2 = nx.Graph(train_A[graph_num].reshape(n,n))
    T = nx.minimum_spanning_tree(G2)
    if(printing): print(sorted(T.edges(data=True)))
    return T

def creating_label(features,y,subject_num,num_node = 20,method='mean_sort',s_num=None):
    features_cluster = [features[(y==(i+1))].numpy().reshape(features.shape[1],-1) for i in range(subject_num)]
    features_cluster = np.array(features_cluster)
    if(s_num==None):
        s_num = random.randint(1,len(features_cluster))-1
    if(method=='cluster'):
        kmeans = KMeans(n_clusters=num_node, random_state=0).fit(features_cluster[s_num])
        label = np.array(kmeans.labels_)
        index = np.zeros((num_node),dtype=int)
        for i in range(num_node):
            index[i] = np.where(label==i)[0][0]
    elif(method=='mean_sort'):
        index = np.argsort(np.mean(features_cluster,axis=2))[s_num]
        index = np.sort(index[(len(index)-num_node):])
    elif(method=='max_sort'):
        index = np.argsort(np.max(features_cluster,axis=2))[s_num]
        index = np.sort(index[(len(index)-num_node):])
    else:
        index = np.sort(random.sample(range(features.shape[1]),num_node))
    return index
        
