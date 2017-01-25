import sys
import tensorflow as tf
import numpy as np

from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer
class AdamaxOptimizer(optimizer.Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, use_locking=False, name="Adamax"):
        super(AdamaxOptimizer, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2

        # Tensor versions of the constructor arguments, created in _prepare().
        self._lr_t = None
        self._beta1_t = None
        self._beta2_t = None

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
        self._beta1_t = ops.convert_to_tensor(self._beta1, name="beta1")
        self._beta2_t = ops.convert_to_tensor(self._beta2, name="beta2")

    def _create_slots(self, var_list):
        # Create slots for the first and second moments.
        for v in var_list:
            self._zeros_slot(v, "m", self._name)
            self._zeros_slot(v, "v", self._name)

    def _apply_dense(self, grad, var):
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
        if var.dtype.base_dtype == tf.float16:
            eps = 1e-7  # Can't use 1e-8 due to underflow -- not sure if it makes a big difference.
        else:
            eps = 1e-8

        v = self.get_slot(var, "v")
        v_t = v.assign(beta1_t * v + (1. - beta1_t) * grad)
        m = self.get_slot(var, "m")
        m_t = m.assign(tf.maximum(beta2_t * m + eps, tf.abs(grad)))
        g_t = v_t / m_t

        var_update = state_ops.assign_sub(var, lr_t * g_t)
        return control_flow_ops.group(*[var_update, m_t, v_t])

    def _apply_sparse(self, grad, var):
        raise NotImplementedError("Sparse gradient updates are not supported.")


class MODEL(object):
    def __init__(self,x_dimension, y_dimension, width, learning_rate,kernNum,parameterPath,batch_size,length_p):
        self.learning_rate = learning_rate
        self.answerNum = 5
        self.WIDTH = width
        self.x_dimension = x_dimension
        self.y_dimension = y_dimension
        
        #self.wordNumberP = wordNumberP
        self.parameterPath = parameterPath    
        self.y_hat = tf.placeholder(shape=(batch_size,self.answerNum), dtype=tf.float32)        
        
        self.p = tf.placeholder(shape=(batch_size,length_p,x_dimension), dtype=tf.float32)    ## (batch,wordNumeberP,x_dimension)    
        self.q = tf.placeholder(shape=(batch_size,None,x_dimension), dtype=tf.float32)    ## (batch,wordNumeberQ,x_dimension)
        self.ansA = tf.placeholder(shape=(batch_size,None,x_dimension), dtype=tf.float32) ## (batch,wordNumeberA,x_dimension)
        self.ansB = tf.placeholder(shape=(batch_size,None,x_dimension), dtype=tf.float32) ## (batch,wordNumeberB,x_dimension)
        self.ansC = tf.placeholder(shape=(batch_size,None,x_dimension), dtype=tf.float32) ## (batch,wordNumeberC,x_dimension)
        self.ansD = tf.placeholder(shape=(batch_size,None,x_dimension), dtype=tf.float32) ## (batch,wordNumeberD,x_dimension)
        self.ansE = tf.placeholder(shape=(batch_size,None,x_dimension), dtype=tf.float32) ## (batch,wordNumeberE,x_dimension)

        with tf.variable_scope('FEATURE_EXTRACTION'):
            wi = tf.get_variable("wi", [width,x_dimension],initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
            bi = tf.get_variable("bi", [width],initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
            wu = tf.get_variable("wu", [width,x_dimension],initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
            bu = tf.get_variable("bu", [width],initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))

        
        with tf.variable_scope('ATTENTION_PARAMETER'):
            wg = tf.get_variable("wg", [width,width],initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
            bg = tf.get_variable("bg", [width],initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))


        with tf.variable_scope('DNN_PARAMETER'):        
            w = tf.get_variable("w", [width,2*width,],initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
            b = tf.get_variable("b", [width],initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))

        wi_batch = tf.tile(tf.expand_dims(wi,0),[batch_size,1,1]) ##(batch,width,x_dimension)
        bi_batch = tf.tile(tf.expand_dims(bi,0),[batch_size,1]) ##(batch,width)

        bi_batch_p = tf.tile(tf.expand_dims(bi_batch,2),[1,1,tf.shape(self.p)[1]])##(batch,width,wordNumeberP)
        bi_batch_q = tf.tile(tf.expand_dims(bi_batch,2),[1,1,tf.shape(self.q)[1]])##(batch,width,wordNumeberQ)
        bi_batch_a = tf.tile(tf.expand_dims(bi_batch,2),[1,1,tf.shape(self.ansA)[1]])##(batch,width,wordNumeberA)
        bi_batch_b = tf.tile(tf.expand_dims(bi_batch,2),[1,1,tf.shape(self.ansB)[1]])##(batch,width,wordNumeberB)
        bi_batch_c = tf.tile(tf.expand_dims(bi_batch,2),[1,1,tf.shape(self.ansC)[1]])##(batch,width,wordNumeberC)
        bi_batch_d = tf.tile(tf.expand_dims(bi_batch,2),[1,1,tf.shape(self.ansD)[1]])##(batch,width,wordNumeberD)
        bi_batch_e = tf.tile(tf.expand_dims(bi_batch,2),[1,1,tf.shape(self.ansE)[1]])##(batch,width,wordNumeberE)
        wu_batch = tf.tile(tf.expand_dims(wu,0),[batch_size,1,1])
        bu_batch = tf.tile(tf.expand_dims(bu,0),[batch_size,1])
        bu_batch_p = tf.tile(tf.expand_dims(bu_batch,2),[1,1,tf.shape(self.p)[1]])##(batch,width,wordNumeberP)
        bu_batch_q = tf.tile(tf.expand_dims(bu_batch,2),[1,1,tf.shape(self.q)[1]])##(batch,width,wordNumeberQ)
        bu_batch_a = tf.tile(tf.expand_dims(bu_batch,2),[1,1,tf.shape(self.ansA)[1]])##(batch,width,wordNumeberA)
        bu_batch_b = tf.tile(tf.expand_dims(bu_batch,2),[1,1,tf.shape(self.ansB)[1]])##(batch,width,wordNumeberB)
        bu_batch_c = tf.tile(tf.expand_dims(bu_batch,2),[1,1,tf.shape(self.ansC)[1]])##(batch,width,wordNumeberC)
        bu_batch_d = tf.tile(tf.expand_dims(bu_batch,2),[1,1,tf.shape(self.ansD)[1]])##(batch,width,wordNumeberD)
        bu_batch_e = tf.tile(tf.expand_dims(bu_batch,2),[1,1,tf.shape(self.ansE)[1]])##(batch,width,wordNumeberE)
        
        ######### NEURAL NETWORK    ######
        featureP = tf.sigmoid(tf.batch_matmul(wi_batch,tf.transpose(self.p,perm =[0,2,1]))+bi_batch_p)*tf.tanh(tf.batch_matmul(wu_batch,tf.transpose(self.p,perm =[0,2,1]))+bu_batch_p) ##(batch,width,wordNumeberP)
        featureQ = tf.sigmoid(tf.batch_matmul(wi_batch,tf.transpose(self.q,perm =[0,2,1]))+bi_batch_q)*tf.tanh(tf.batch_matmul(wu_batch,tf.transpose(self.q,perm =[0,2,1]))+bu_batch_q) ##(batch,width,wordNumeberQ)
        featureA = tf.sigmoid(tf.batch_matmul(wi_batch,tf.transpose(self.ansA,perm =[0,2,1]))+bi_batch_a)*tf.tanh(tf.batch_matmul(wu_batch,tf.transpose(self.ansA,perm =[0,2,1]))+bu_batch_a) ##(batch,width,wordNumeberA)
        featureB = tf.sigmoid(tf.batch_matmul(wi_batch,tf.transpose(self.ansB,perm =[0,2,1]))+bi_batch_b)*tf.tanh(tf.batch_matmul(wu_batch,tf.transpose(self.ansB,perm =[0,2,1]))+bu_batch_b) ##(batch,width,wordNumeberB)
        featureC = tf.sigmoid(tf.batch_matmul(wi_batch,tf.transpose(self.ansC,perm =[0,2,1]))+bi_batch_c)*tf.tanh(tf.batch_matmul(wu_batch,tf.transpose(self.ansC,perm =[0,2,1]))+bu_batch_c) ##(batch,width,wordNumeberC)
        featureD = tf.sigmoid(tf.batch_matmul(wi_batch,tf.transpose(self.ansD,perm =[0,2,1]))+bi_batch_d)*tf.tanh(tf.batch_matmul(wu_batch,tf.transpose(self.ansD,perm =[0,2,1]))+bu_batch_d) ##(batch,width,wordNumeberD)
        featureE = tf.sigmoid(tf.batch_matmul(wi_batch,tf.transpose(self.ansE,perm =[0,2,1]))+bi_batch_e)*tf.tanh(tf.batch_matmul(wu_batch,tf.transpose(self.ansE,perm =[0,2,1]))+bu_batch_e) ##(batch,width,wordNumeberE)


        wg_batch = tf.tile(tf.expand_dims(wg,0),[batch_size,1,1]) ##(batch,width,width)
        bg_batch = tf.tile(tf.expand_dims(bg,0),[batch_size,1]) ##(batch,width)
        bg_batch_q = tf.tile(tf.expand_dims(bg_batch,2),[1,1,tf.shape(featureQ)[2]])##(batch,width,wordNumeberQ)
        bg_batch_a = tf.tile(tf.expand_dims(bg_batch,2),[1,1,tf.shape(featureA)[2]])##(batch,width,wordNumeberA)
        bg_batch_b = tf.tile(tf.expand_dims(bg_batch,2),[1,1,tf.shape(featureB)[2]])##(batch,width,wordNumeberB)
        bg_batch_c = tf.tile(tf.expand_dims(bg_batch,2),[1,1,tf.shape(featureC)[2]])##(batch,width,wordNumeberC)
        bg_batch_d = tf.tile(tf.expand_dims(bg_batch,2),[1,1,tf.shape(featureD)[2]])##(batch,width,wordNumeberD)
        bg_batch_e = tf.tile(tf.expand_dims(bg_batch,2),[1,1,tf.shape(featureE)[2]])##(batch,width,wordNumeberE)

        attentionQ = self.softmax_col(tf.batch_matmul(tf.transpose(tf.batch_matmul(wg_batch,featureQ)+bg_batch_q,perm=[0,2,1]),featureP))  ##(batch,wordNumberQ, wordNumberP)
        attentionA = self.softmax_col(tf.batch_matmul(tf.transpose(tf.batch_matmul(wg_batch,featureA)+bg_batch_a,perm=[0,2,1]),featureP))  ##(batch,wordNumberA, wordNumberP)
        attentionB = self.softmax_col(tf.batch_matmul(tf.transpose(tf.batch_matmul(wg_batch,featureB)+bg_batch_b,perm=[0,2,1]),featureP))  ##(batch,wordNumberB, wordNumberP)
        attentionC = self.softmax_col(tf.batch_matmul(tf.transpose(tf.batch_matmul(wg_batch,featureC)+bg_batch_c,perm=[0,2,1]),featureP))  ##(batch,wordNumberC, wordNumberP)
        attentionD = self.softmax_col(tf.batch_matmul(tf.transpose(tf.batch_matmul(wg_batch,featureD)+bg_batch_d,perm=[0,2,1]),featureP))  ##(batch,wordNumberD, wordNumberP)
        attentionE = self.softmax_col(tf.batch_matmul(tf.transpose(tf.batch_matmul(wg_batch,featureE)+bg_batch_e,perm=[0,2,1]),featureP))  ##(batch,wordNumberE, wordNumberP)
    
        
        hiddenQ = tf.batch_matmul(featureQ,attentionQ)    #(batch,width, wordNumberP)
        hiddenA = tf.batch_matmul(featureA,attentionA)    #(batch,width, wordNumberP)
        hiddenB = tf.batch_matmul(featureB,attentionB)    #(batch,width, wordNumberP)
        hiddenC = tf.batch_matmul(featureC,attentionC)    #(batch,width, wordNumberP)
        hiddenD = tf.batch_matmul(featureD,attentionD)    #(batch,width, wordNumberP)
        hiddenE = tf.batch_matmul(featureE,attentionE)    #(batch,width, wordNumberP)


        w_batch = tf.tile(tf.expand_dims(w,0),[batch_size,1,1]) ##(batch,width,2*width)
        b_batch = tf.tile(tf.expand_dims(b,0),[batch_size,1]) ##(batch,width)
        b_batch_p = tf.tile(tf.expand_dims(b_batch,2),[1,1,tf.shape(featureP)[2]])##(batch,width,wordNumberP)
        

        comparedQ = tf.nn.relu( tf.batch_matmul(w_batch,self.SubMulti(featureP,hiddenQ))+b_batch_p)    #(batch,width, wordNumberP)
        comparedA = tf.nn.relu( tf.batch_matmul(w_batch,self.SubMulti(featureP,hiddenA))+b_batch_p)    #(batch,width, wordNumberP)
        comparedB = tf.nn.relu( tf.batch_matmul(w_batch,self.SubMulti(featureP,hiddenB))+b_batch_p)    #(batch,width, wordNumberP)
        comparedC = tf.nn.relu( tf.batch_matmul(w_batch,self.SubMulti(featureP,hiddenC))+b_batch_p)    #(batch,width, wordNumberP)
        comparedD = tf.nn.relu( tf.batch_matmul(w_batch,self.SubMulti(featureP,hiddenD))+b_batch_p)    #(batch,width, wordNumberP)
        comparedE = tf.nn.relu( tf.batch_matmul(w_batch,self.SubMulti(featureP,hiddenE))+b_batch_p)    #(batch,width, wordNumberP)



        #### start different  ###
        comparedQA = tf.expand_dims(tf.transpose(tf.concat(1,[comparedQ,comparedA]),perm = [0,2,1]),-1) #(batch,wordNumberP,2width,1)
        comparedQB = tf.expand_dims(tf.transpose(tf.concat(1,[comparedQ,comparedB]),perm = [0,2,1]),-1) #(batch,wordNumberP,2width,1)
        comparedQC = tf.expand_dims(tf.transpose(tf.concat(1,[comparedQ,comparedC]),perm = [0,2,1]),-1) #(batch,wordNumberP,2width,1)
        comparedQD = tf.expand_dims(tf.transpose(tf.concat(1,[comparedQ,comparedD]),perm = [0,2,1]),-1) #(batch,wordNumberP,2width,1)
        comparedQE = tf.expand_dims(tf.transpose(tf.concat(1,[comparedQ,comparedE]),perm = [0,2,1]),-1) #(batch,wordNumberP,2width,1)


        pooled_outputs_QA = []
        pooled_outputs_QB = []
        pooled_outputs_QC = []
        pooled_outputs_QD = []
        pooled_outputs_QE = []
        filter_sizes = [1,3,5]
        num_filters = 128
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                filter_shape = [filter_size, x_dimension, 1, num_filters]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
            # Convolution Layer
            convQA = tf.nn.conv2d(
                comparedQA,
                W,
                strides=[1, 1, 1, 1],
                padding="VALID",
                name="conv")## [batch,wordNumberP- filter_size + 1, 1, 1]

            convQB = tf.nn.conv2d(
                comparedQB,
                W,
                strides=[1, 1, 1, 1],
                padding="VALID",
                name="conv")## [batch,wordNumberP- filter_size + 1, 1, 1]

            convQC = tf.nn.conv2d(
                comparedQC,
                W,
                strides=[1, 1, 1, 1],
                padding="VALID",
                name="conv")## [batch,wordNumberP- filter_size + 1, 1, 1]


            convQD = tf.nn.conv2d(
                comparedQD,
                W,
                strides=[1, 1, 1, 1],
                padding="VALID",
                name="conv")## [batch,wordNumberP- filter_size + 1, 1, 1]

            convQE = tf.nn.conv2d(
                comparedQE,
                W,
                strides=[1, 1, 1, 1],
                padding="VALID",
                name="conv")## [batch,wordNumberP- filter_size + 1, 1, 1]


            hQA = tf.nn.relu(tf.nn.bias_add(convQA, b), name="relu")
            hQB = tf.nn.relu(tf.nn.bias_add(convQB, b), name="relu")
            hQC = tf.nn.relu(tf.nn.bias_add(convQC, b), name="relu")
            hQD = tf.nn.relu(tf.nn.bias_add(convQD, b), name="relu")
            hQE = tf.nn.relu(tf.nn.bias_add(convQE, b), name="relu")
                          

            # Max-pooling over the outputs
            pooledQA = tf.nn.max_pool(
                hQA,
                ksize=[1, length_p - filter_size + 1, 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="pool")  ##  [batch_size, 1, 1, num_filters]

            pooled_outputs_QA.append(pooledQA)

            pooledQB = tf.nn.max_pool(
                hQB,
                ksize=[1, length_p - filter_size + 1, 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="pool")  ##[batch_size, 1, 1, num_filters]
            pooled_outputs_QB.append(pooledQB)
            pooledQC = tf.nn.max_pool(
                hQC,
                ksize=[1, length_p - filter_size + 1, 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="pool")  ##[batch_size, 1, 1, num_filters]
            pooled_outputs_QC.append(pooledQC)
            pooledQD = tf.nn.max_pool(
                hQD,
                ksize=[1, length_p - filter_size + 1, 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="pool")  ##[batch_size, 1, 1, num_filters]
            pooled_outputs_QD.append(pooledQD)
            pooledQE = tf.nn.max_pool(
                hQE,
                ksize=[1, length_p - filter_size + 1, 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="pool")  ##[batch_size, 1, 1, num_filters]
            pooled_outputs_QE.append(pooledQE)

 
        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        h_pool_QA = tf.concat(3, pooled_outputs_QA) ##[batch_size, 1, 1,num_filters_total]
        h_pool_QB = tf.concat(3, pooled_outputs_QB) ##[batch_size, 1, 1,num_filters_total]
        h_pool_QC = tf.concat(3, pooled_outputs_QC) ##[batch_size, 1, 1,num_filters_total]
        h_pool_QD = tf.concat(3, pooled_outputs_QD) ##[batch_size, 1, 1,num_filters_total]
        h_pool_QE = tf.concat(3, pooled_outputs_QE) ##[batch_size, 1, 1,num_filters_total]

        h_pool_flat_QA = tf.expand_dims(tf.reshape(h_pool_QA, [-1, num_filters_total]),2) #[batch_size, num_filters_total,1]
        h_pool_flat_QB = tf.expand_dims(tf.reshape(h_pool_QB, [-1, num_filters_total]),2) #[batch_size, num_filters_total,1]
        h_pool_flat_QC = tf.expand_dims(tf.reshape(h_pool_QC, [-1, num_filters_total]),2) #[batch_size, num_filters_total,1]
        h_pool_flat_QD = tf.expand_dims(tf.reshape(h_pool_QD, [-1, num_filters_total]),2) #[batch_size, num_filters_total,1]
        h_pool_flat_QE = tf.expand_dims(tf.reshape(h_pool_QE, [-1, num_filters_total]),2) #[batch_size, num_filters_total,1]

        result = tf.concat(2,[h_pool_flat_QA,h_pool_flat_QB,h_pool_flat_QC,h_pool_flat_QD,h_pool_flat_QE])  #[batch_size, num_filters_total,5]

        with tf.variable_scope('last_DNN_PARAMETER'):

            wtrans = tf.get_variable("wtrans", [width,num_filters_total],initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))    
            btrans = tf.get_variable("btrans", [width],initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))    
            wout = tf.get_variable("wout", [1,width,],initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))    
            bout = tf.get_variable("bout", [1],initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))    

        wtrans_batch = tf.tile(tf.expand_dims(wtrans,0),[batch_size,1,1])
        btrans_batch = tf.tile(tf.expand_dims(btrans,0),[batch_size,1]) ##(batch,width)
        btrans_batch = tf.tile(tf.expand_dims(btrans_batch,2),[1,1,tf.shape(result)[2]])##(batch,width,5)

        y = tf.tanh(tf.batch_matmul(wtrans_batch,result)+btrans_batch) ##(batch,width,5)

        wout_batch = tf.tile(tf.expand_dims(wout,0),[batch_size,1,1]) ## (batch,1,width)
        bout_batch = tf.tile(tf.expand_dims(bout,0),[batch_size,1]) ##(batch,1)
        bout_batch = tf.tile(tf.expand_dims(bout_batch,2),[1,1,tf.shape(y)[2]]) ## ##(batch,1,5)

        

        y = tf.squeeze(tf.batch_matmul(wout_batch,y)+bout) #[batch_size,5]

        self.predict_result = tf.argmax(y, dimension=1)
        
        self.cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(y,self.y_hat))

        #rate = tf.train.exponential_decay(lr ,global_step,decay_step,decay_rate)
        global_step = tf.Variable(0, trainable=False)
        self.train_op = AdamaxOptimizer(learning_rate = 0.002).minimize(self.cost,global_step=global_step)

        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement = True,log_device_placement = True))


    def train(self,input_p,input_q,input_ansA,input_ansB,input_ansC,input_ansD,input_ansE,input_y_hat):

        print(input_q.shape)
        _,loss,predict = self.sess.run([self.train_op,self.cost,self.predict_result],feed_dict={self.p:input_p,self.q:input_q,self.ansA:input_ansA,
                                                                    self.ansB:input_ansB,self.ansC:input_ansC,self.ansD:input_ansD,
                                                                    self.ansE:input_ansE,self.y_hat:input_y_hat})
        
        return loss,predict
    
    def initialize(self):
        print("init")
        self.sess.run(tf.initialize_all_variables())

    def softmax_col(self,tensor):

        total = tf.reduce_sum(tf.exp(tensor),1) ##(tensor(0),tensor(2))
        total = tf.tile(tf.expand_dims(total,1),[1,tf.shape(tensor)[1],1])  ##(tensor(0),tensor(1),tensor(2))

        return tf.exp(tensor)/total

    def SubMulti(self, a, h):
        
        return tf.concat(1,[(a-h)*(a-h), a*h]) ## (batch,2*width,worldNumberP)

    def predict(self,input_p,input_q,input_ansA,input_ansB,input_ansC,input_ansD,input_ansE):
        
        return self.sess.run(self.predict_result,feed_dict={self.p:input_p,self.q:input_q,self.ansA:input_ansA,
                                                    self.ansB:input_ansB,self.ansC:input_ansC,self.ansD:input_ansD,
                                                    self.ansE:input_ansE})
        





