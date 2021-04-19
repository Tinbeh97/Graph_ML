import numpy as np
import networkx.algorithms as nl
import networkx as nx

def Degree(M):
    if(M.ndim==3):
        d = np.sum(M, axis=2)
        D = np.zeros((d.shape[0], d.shape[1]*d.shape[1]))
        D[:, ::d.shape[1]+1] = d
        D = D.reshape(d.shape[0],d.shape[1],d.shape[1])
        for i in range(d.shape[1]):
            D[:,i,i] = d[:,i]
    else:
        D = np.diag(np.sum(M,axis=1)) 
    return D

def global_eff(G):
    sp = dict(nl.shortest_paths.generic.shortest_path_length(G,weight='weight'))
    n = len(sp)
    n_eff = np.zeros(n)
    for i in range(n):
        sps = sp; sps[i][i] = 1000
        spi = (np.array(list(sps[i].values())))
        n_eff[i] = 1 / np.min(spi)
    g_eff = np.mean(n_eff)/(n-1)

    return np.mean(g_eff)

def matrix_feature(M,n,connected):
    features = np.zeros((M.shape[0],6+n))
    for i in range(M.shape[0]):
        G = nx.Graph(M[i])
        #degree = np.sum(M[i],axis=1)
        #betw = np.array(list(nx.betweenness_centrality(G,weight='weight').values()))
        pr = np.array(list(nx.pagerank(G,weight='weight').values()))
        #clu = np.array(list(nx.clustering(G,weight='weight').values()))
        #eig = np.array(list(nx.eigenvector_centrality(G,weight='weight').values()))
        #nodal_par = np.array([degree,betw,pr,clu,eig]).reshape(-1)
        
        #transitivity
        tra = nl.cluster.transitivity(G)
        #modularity
        di_M = M[i]>0
        k = np.sum(di_M,axis=1)
        l = np.sum(k)
        mod = np.sum(np.multiply((M[i] - (k.T * k)/l),di_M)) / l 
        if(connected==1):
            #path length
            path = nl.shortest_paths.generic.average_shortest_path_length(G,weight='weight')
            #diameter
            dia = nl.distance_measures.diameter(G)
            #radius
            ra = nl.distance_measures.radius(G)
        else:
            if(nx.is_connected(G)):
                path = nl.shortest_paths.generic.average_shortest_path_length(G,weight='weight')
                ra = nl.distance_measures.radius(G)
                dia = nl.distance_measures.diameter(G)
            else:
                path = 10000
                dia = 10000
                ra = 10000
                """
                for g in nx.connected_component_subgraphs(G):
                    path = np.maximum(nl.shortest_paths.generic.average_shortest_path_length(g),path)
                    ra = np.maximum(nl.distance_measures.radius(g),ra)
                    dia = np.maximum(nl.distance_measures.diameter(g),dia)
                """
        #global efficiency
        gf = global_eff(G)
        global_par = np.array([tra,mod,path,gf,dia,ra]).reshape(-1)
        features[i,:] = np.concatenate((global_par,pr))
    return features

def node_features(M,n):
    features = np.zeros((M.shape[0],n,2))
    for i in range(M.shape[0]):
        G = nx.Graph(M[i])
        degree = np.sum(M[i],axis=1)
        clu = np.array(list(nx.clustering(G,weight='weight').values()))
        """
        betw = np.array(list(nx.betweenness_centrality(G,weight='weight').values()))
        pr = np.array(list(nx.pagerank(G,weight='weight').values()))
        eig = np.array(list(nx.eigenvector_centrality(G,weight='weight').values()))
        """
        features[i,:] = np.array([degree,clu]).reshape(n,-1)
    return features

def graph_norm(adj):
    I = (np.tile(np.eye(adj.shape[1]),adj.shape[0]).T).reshape(-1,adj.shape[1],adj.shape[1])
    adj_ = adj + I
    d = np.sum(adj_,axis=2)**(-0.5)
    D_inv = np.zeros((d.shape[0], d.shape[1]*d.shape[1]))
    D_inv[:, ::d.shape[1]+1] = d
    D_inv = D_inv.reshape(d.shape[0],d.shape[1],d.shape[1])
    adj_normalized = np.einsum('ijk,ikm->ijm',adj,D_inv)
    adj_normalized = np.einsum('ijk,ikm->ijm',D_inv,adj_normalized)
    return np.round(adj_normalized,8)
