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
