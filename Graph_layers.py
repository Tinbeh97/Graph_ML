import tensorflow as tf
import tensorflow.keras.backend as bk
from clustering import graph_clustering
import numpy as np
from copy import deepcopy
        
Type = "float32"

class GraphConvolution(tf.keras.layers.Layer):
    """ Graph convolution layer """
    def __init__(self, input_dim, output_dim, num, act = tf.nn.relu, **kwargs):
        super().__init__(**kwargs)
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable( name = 'weight'+str(num),
            initial_value=w_init(shape=(input_dim, output_dim), dtype=Type),
            trainable=True)
        b_init = tf.zeros_initializer()
        self.b = tf.Variable( name = 'bias'+str(num),
            initial_value=b_init(shape=(output_dim,), dtype=Type), trainable=True)
        self.act = act

    def call(self, inputs, adj, rate =0., normalize=False):
        x = tf.nn.dropout(inputs, rate = rate)
        x = tf.matmul(x, self.w)
        x = tf.matmul(adj, x)
        outputs = self.act(x + self.b)
        if normalize:
            x = tf.keras.utils.normalize(x)
        return outputs
    
class GraphLinear(tf.keras.layers.Layer):
    """ Graph linear layer """
    def __init__(self, input_dim, output_dim, num, act = tf.nn.relu, **kwargs):
        super().__init__(**kwargs)
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable( name = 'weight'+str(num),
            initial_value=w_init(shape=(input_dim, output_dim), dtype=Type),
            trainable=True)
        b_init = tf.zeros_initializer()
        self.b = tf.Variable( name = 'bias'+str(num),
            initial_value=b_init(shape=(output_dim,), dtype=Type), trainable=True)
        self.act = act
        
    def call(self, inputs, normalize=False):
        x = tf.matmul(inputs, self.w)
        outputs = self.act(x + self.b)
        if normalize:
            x = tf.keras.utils.normalize(x)
        return outputs
    
class Graph_diffpool(tf.keras.layers.Layer):
    """ Graph diff pooling layer """
    def __init__(self, input_dim, output_dim, num, act = tf.nn.relu, **kwargs):
        super().__init__(**kwargs)
        self.h = GraphConvolution(input_dim = input_dim, 
                                  output_dim = output_dim, num = num,
                                  act = act)
        
    def call(self, inputs, adj, rate, normalize=False):
        S = self.h(inputs, adj, rate, normalize)
        S = tf.nn.softmax(S,axis=-1)
        S_T = tf.transpose(S, perm=[0, 2, 1])
        #loss
        LP_loss = adj - tf.matmul(S,S_T)
        LP_loss = tf.reduce_mean(tf.norm(LP_loss, axis=(-1, -2)))
        self.add_loss(LP_loss)
        entr = tf.negative(tf.reduce_sum(tf.multiply(S, bk.log(S + bk.epsilon())), axis=-1))
        entr_loss = tf.reduce_mean(entr)
        self.add_loss(entr_loss)
        #new_output
        x = tf.matmul(S_T,inputs)
        adj = tf.matmul(adj,S)
        adj = tf.matmul(S_T,adj)
        return x, adj
    
class Graph_sagepool(tf.keras.layers.Layer):
    """ Graph sage pooling layer """
    def __init__(self, input_dim, num, ratio, act = tf.nn.relu, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio
        self.h = GraphConvolution(input_dim = input_dim, 
                                  output_dim = 1, num = num,
                                  act = act)
        
    def call(self, inputs, adj, rate, normalize=False):
        K = int(self.ratio * adj.shape[1])
        n = adj.shape[1]
        num_nodes = n-K
        y = self.h(inputs, adj, rate, normalize)
        y = tf.reshape(y,[-1,adj.shape[1]])
        y = tf.math.tanh(y)
        indices = tf.argsort(y,axis=-1)
        indices = indices[:,K:]
        u = tf.repeat(tf.reshape(tf.range(len(indices)),(-1,1)),indices.shape[1],axis=1)
        index = tf.concat([tf.reshape(u,[-1,1]),tf.reshape(indices,[-1,1])],1)
        mask = tf.scatter_nd(index, tf.reshape(tf.ones_like(indices),-1), tf.constant([len(adj),n]))
        x = tf.boolean_mask(inputs,mask)
        x = tf.math.multiply(x , tf.boolean_mask(tf.expand_dims(y,axis=2),mask))
        x = tf.reshape(x,[-1,num_nodes,x.shape[1]])
        adj = tf.boolean_mask(adj,mask,axis=0)
        adj = tf.reshape(adj,[-1,num_nodes,n])
        adj = tf.transpose(adj, perm=[0, 2, 1])
        adj = tf.boolean_mask(adj,mask,axis=0)
        adj = tf.reshape(adj,[-1,num_nodes,num_nodes])
        return x, adj
    
class Graph_globalpool(tf.keras.layers.Layer):
    """ Graph global pooling layer"""
    def __init__(self,pool_method='max',**kwargs):
        super().__init__(**kwargs)
        self.method = pool_method
    def call(self, inputs):
        if(self.method=='max'):
            return tf.reduce_max(inputs,axis=-1)
        elif(self.method=='mean'):
            return tf.reduce_mean(inputs,axis=-1)
        elif(self.method=='sum'):
            return tf.reduce_sum(inputs,axis=-1)
        
class Graph_clustpool_2(tf.keras.layers.Layer):
    """ Graph clustering pooling layer """
    def __init__(self, adj, ratio, act = lambda x: x, **kwargs):
        super().__init__(**kwargs)
        self.n = adj.shape[1]
        self.act = act
        self.cluster_labels = graph_clustering(adj,'kmeans',2,ratio=ratio).reshape(1,-1)
        self.n_cluster = np.sum(self.cluster_labels)
    def adj_masking(self,adj):
        adj = tf.cast(adj,Type)
        mask = tf.repeat(self.cluster_labels,len(adj),axis=0)
        adj = tf.boolean_mask(adj,mask,axis=0)
        adj = tf.reshape(adj,[-1,self.n_cluster,self.n])
        adj = tf.transpose(adj, perm=[0, 2, 1])
        adj = tf.boolean_mask(adj,mask,axis=0)
        adj = tf.reshape(adj,[-1,self.n_cluster,self.n_cluster])
        return adj
    def call(self, inputs, adj, normalize=False):
        mask = tf.repeat(self.cluster_labels,len(adj),axis=0)
        x = tf.boolean_mask(inputs,mask)
        x = tf.reshape(x,[-1,self.n_cluster,x.shape[1]])
        return x, adj

class Graph_clustpool(tf.keras.layers.Layer):
    """ Graph clustering pooling layer """
    def __init__(self, adj, n_cluster, cluster_type = 'sum', act = lambda x: x, **kwargs):
        super().__init__(**kwargs)
        self.n_cluster = n_cluster
        self.cluster_type = cluster_type
        self.act = act
        self.adj = adj
        self.cluster_labels = graph_clustering(self.adj,'kmeans',self.n_cluster)
        mask = np.zeros((self.adj.shape[1],self.n_cluster))
        for i in range(self.n_cluster):
            mask[:,i] = np.equal(self.cluster_labels,i)
        self.mask = tf.cast(mask,dtype=Type)
    def adj_masking(self,adj):
        if(self.cluster_type=='sum'):
            adj = tf.einsum('ijk,kn->ijn',adj,self.mask)
            adj = tf.einsum('ijk,jn->ink',adj,self.mask)
        else:
            all_adj = tf.einsum('nij,ik->nikj',adj,self.mask)
            if(self.cluster_type=='max'):
                all_adj = tf.math.reduce_max(all_adj,axis=1)
            elif(self.cluster_type=='mean'):
                all_adj = tf.math.reduce_mean(all_adj,axis=1) 
            all_adj = tf.einsum('nij,jk->nijk',all_adj,self.mask)
            if(self.cluster_type=='max'):
                adj = tf.math.reduce_max(all_adj,axis=2)
            elif(self.cluster_type=='mean'):
                adj = tf.math.reduce_mean(all_adj,axis=2) 
        return adj
    #pytorch masking
    def masking(self, inputs, adj, labels):
        import torch
        from torch_scatter import scatter
        labels = torch.tensor(labels).type(torch.LongTensor)
        x = scatter(torch.tensor(inputs),labels,dim=1, reduce="mean")
        adj = scatter(torch.tensor(adj),labels,dim=-1, reduce="mean")
        adj = scatter(adj,labels,dim=1, reduce="mean")
        return tf.cast(x.numpy(),tf.float32), tf.cast(adj.numpy(),tf.float32)
    #@tf.function
    def call(self, inputs, adj, normalize=False):
        #x, adj = self.masking(inputs.numpy(),adj.numpy(),self.cluster_labels)
        if(self.cluster_type=='sum'):
            x = tf.einsum('ijk,jn->ink',inputs,self.mask)
        else:
            all_x = tf.einsum('nij,ik->nikj',inputs,self.mask)
            if(self.cluster_type=='max'):
                x = tf.math.reduce_max(all_x,axis=1)
            elif(self.cluster_type=='mean'):
                x = tf.math.reduce_mean(all_x,axis=1)

        """
        in2 = inputs.numpy()
        x2 = np.zeros((inputs.shape[0],self.n_cluster,inputs.shape[2]))
        adj2 = deepcopy(adj.numpy())
        cl = deepcopy(cluster_labels)
        for i in range(self.n_cluster):
            labels = (cl==i)
            x2[:,i,:] = np.sum(in2[:,(cluster_labels==i),:],axis=1)
            u1 = adj2[:,labels,:]
            u2 = u1[:,:,labels]
            u3 = adj2[:,~labels,:]
            u3 = u3[:,:,~labels]
            m = np.insert(u3, i, np.mean(u1[:,:,~labels],axis=1), axis=1)
            m = np.insert(m, i, 1, axis=2)
            m[:,:,i] = m[:,i,:]
            I = (np.tile(np.eye(u2.shape[1]),u2.shape[0]).T).reshape(-1,u2.shape[1],u2.shape[1])
            m[:,i,i] = np.sum(np.multiply(u2,I),axis=(1,2))
            adj2 = deepcopy(m)
            cl = np.delete(cl,np.where(cl==i))
            cl = np.insert(cl,i,i)
        adj , x = tf.cast(adj2,dtype=tf.float32), tf.cast(x2,dtype=tf.float32)
        """
        """
        in2 = inputs.numpy()
        x2 = np.zeros((inputs.shape[0],self.n_cluster,inputs.shape[2]))
        adj12 = adj.numpy()
        adj2 = np.zeros((inputs.shape[0],self.n_cluster,self.n_cluster))
        for i in range(self.n_cluster):
            cl = (cluster_labels==i)
            x2[:,i,:] = np.sum(in2[:,cl,:],axis=1)
            u1 = adj12[:,labels,:]
            u2 = u1[:,:,labels]
            I = (np.tile(np.eye(u2.shape[1]),u2.shape[0]).T).reshape(-1,u2.shape[1],u2.shape[1])
            adj2[:,i,i] = np.sum(np.multiply(u2,I),axis=(1,2))
            
            u3 = adj2[:,~labels,:]
            u3 = u3[:,:,~labels]
            m = np.insert(u3, i, np.mean(u1[:,:,~labels],axis=1), axis=1)
            m = np.insert(m, i, 1, axis=2)
            m[:,:,i] = m[:,i,:]
        adj , x = tf.cast(adj2,dtype=tf.float32), tf.cast(x2,dtype=tf.float32)
        """
        return x, adj
        
class InnerProductDecoder(tf.keras.layers.Layer):
    """Symmetric inner product decoder layer"""
    def __init__(self , act = tf.nn.sigmoid, **kwargs):
        super().__init__(**kwargs)
        self.act = act

    def call(self, inputs, rate = 0.):
        inputs = tf.nn.dropout(inputs, rate = rate)
        if (tf.shape(inputs).shape==3):
            x = tf.transpose(inputs, perm=[0, 2, 1])
        else:
            x = tf.transpose(inputs)
        x = tf.matmul(inputs, x)
        """
        if (tf.shape(inputs).shape==3):
            x = tf.reshape(x, [-1,x.shape[1]*x.shape[2]])
        else:
            x = tf.reshape(x, [-1])
        """
        outputs = self.act(x)
        return outputs