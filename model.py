import tensorflow as tf
import numpy as np

def save_weights(saver, sess, path='saves/weights.ckpt'):
    saver.save(sess, path)


def load_weights(saver, sess, path='saves/weights.ckpt'):
    saver.restore(sess, path)


def linear(x, size, name, initializer=None, bias_init=0):
    w = tf.get_variable(name + "/w", [x.get_shape()[1], size], initializer=initializer)
    b = tf.get_variable(name + "/b", [size], initializer=tf.constant_initializer(bias_init))
    return tf.matmul(x, w) + b

def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

def flatten(x):
    return tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])])


def conv2d(x, num_filters, name, filter_size=(3, 3), stride=(1, 1), pad="SAME", dtype=tf.float32, collections=None):
    with tf.variable_scope(name):
        stride_shape = [1, stride[0], stride[1], 1]
        filter_shape = [filter_size[0], filter_size[1], int(x.get_shape()[3]), num_filters]

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[:3])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = np.prod(filter_shape[:2]) * num_filters
        # initialize weights with random weights
        w_bound = np.sqrt(6. / (fan_in + fan_out))

        w = tf.get_variable("W", filter_shape, dtype, tf.random_uniform_initializer(-w_bound, w_bound),
                            collections=collections)
        b = tf.get_variable("b", [1, 1, 1, num_filters], initializer=tf.constant_initializer(0.0),
                            collections=collections)
        return tf.nn.conv2d(x, w, stride_shape, pad) + b

def universeHead(x, nConvs=4):
    ''' universe agent example
        input: [None, 42, 42, 1]; output: [None, 288];
    '''
    print('Using universe head design')
    for i in range(nConvs):
        x = tf.nn.elu(conv2d(x, 32, "l{}".format(i + 1), (3, 3), (2, 2)))
        if i == 0:
            convout = x[1, :, :, 0:16]
        # print('Loop{} '.format(i+1),tf.shape(x))
        # print('Loop{}'.format(i+1),x.get_shape())
    cs = convout.get_shape()
    convout = tf.reshape(convout,[1,cs[0].value,cs[1].value,16])
    convout = tf.transpose(convout,perm=[3,1,2,0])
    print(convout.get_shape())
    x = flatten(x)
    print(x.get_shape())
    return convout, x

def doomHead(x):
    ''' Learning by Prediction ICLR 2017 paper
        (their final output was 64 changed to 256 here)
        input: [None, 120, 160, 1]; output: [None, 1280] -> [None, 256];
    '''
    print('Using doom head design')
    x = tf.nn.elu(conv2d(x, 8, "l1", [5, 5], [4, 4]))
    x = tf.nn.elu(conv2d(x, 16, "l2", [3, 3], [2, 2]))
    x = tf.nn.elu(conv2d(x, 32, "l3", [3, 3], [2, 2]))
    x = tf.nn.elu(conv2d(x, 64, "l4", [3, 3], [2, 2]))
    x = flatten(x)
    x = tf.nn.elu(linear(x, 256, "fc", normalized_columns_initializer(0.01)))
    return x

class StateActionPredictor(object):
    def __init__(self, ob_space, ac_space, load_from_file=False):

        self.beta = 0.5

        # input: s1,s2: : [None, h, w, ch] (usually ch=1 or 4)
        # asample: 1-hot encoding of sampled action from policy: [None, ac_space]
        input_shape = [None] + list(ob_space)
        self.s1 = phi1 = tf.placeholder(tf.float32, input_shape)
        self.s2 = phi2 = tf.placeholder(tf.float32, input_shape)
        self.asample = tf.placeholder(tf.float32, [None, ac_space])

        # feature encoding: phi1, phi2: [None, LEN]
        size = 256

        self.convout, phi1 = universeHead(phi1)
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            self.convout2, phi2 = universeHead(phi2)
        phisize = phi1.get_shape()[1]
        print(int(phisize))
        print(self.convout)
        print(self.convout.get_shape())
        # inverse model: g(phi1,phi2) -> a_inv: [None, ac_space]
        g = tf.concat([phi1, phi2], 1)
        g = tf.layers.dense(g, size, tf.nn.relu)#tf.nn.relu(linear(g, size, "g1", normalized_columns_initializer(0.01)))
        aindex = tf.argmax(self.asample, axis=1)  # aindex: [batch_size,]
        self.logits = tf.layers.dense(g, ac_space, tf.nn.relu)#self.logits = linear(g, ac_space, "glast", normalized_columns_initializer(0.01))
        """tf.nn.sparse_softmax_cross_entropy_with_logits(
                                        logits=self.logits, labels=aindex), name="invloss") * (int(phisize) / 288.0)"""

        self.invloss = tf.reduce_mean(tf.square(tf.subtract(self.logits, self.asample))) * 10
        self.mat = tf.nn.sparse_softmax_cross_entropy_with_logits(
                                        logits=self.logits, labels=aindex)
        self.mat2 = self.logits
        self.ainvprobs = tf.nn.softmax(self.logits, axis=-1)

        # forward model: f(phi1,asample) -> phi2
        # Note: no backprop to asample of policy: it is treated as fixed for predictor training
        f = tf.concat([phi1, self.asample],1)
        f = tf.layers.dense(f,size,tf.nn.relu)  # f = tf.nn.relu(linear(f, size, "f1", normalized_columns_initializer(0.01)))
        f = tf.layers.dense(f,phi1.get_shape()[1].value,tf.nn.sigmoid)  # f = linear(f, phi1.get_shape()[1].value, "flast", normalized_columns_initializer(0.01))

        self.forwardloss = 0.5 * tf.reduce_mean(tf.square(tf.subtract(f, phi2)), name='forwardloss')
        # self.forwardloss = 0.5 * tf.reduce_mean(tf.sqrt(tf.abs(tf.subtract(f, phi2))), name='forwardloss')
        # self.forwardloss = cosineLoss(f, phi2, name='forwardloss')
        self.forwardloss = self.forwardloss * 100.0  # lenFeatures=288. Factored out to make hyperparams not depend on it.

        self.totalloss = tf.add(self.forwardloss*self.beta, self.invloss *(1-self.beta))
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(self.invloss)

        self.saver = tf.train.Saver()

        self.sess = tf.Session()
        if load_from_file:
            load_weights(self.saver,self.sess)
        else:
            self.sess.run(tf.global_variables_initializer())
        self.init_summarizer()

    def init_summarizer(self):
        tf.summary.histogram("phi1", self.convout)
        tf.summary.histogram("phi2", self.convout2)
        self.merge = tf.summary.merge_all()
        self.file = tf.summary.FileWriter('visualizer', self.sess.graph)

    def sumarize(self, summary, step):
        self.file.reopen()
        self.file.add_summary(summary, step)
        self.file.close()

    def save_weights(self):
        save_weights(self.saver, self.sess)

    def train(self, s1, s2, asample, step):
        error = self.sess.run([self.merge, self.optimizer, self.totalloss,self.forwardloss,self.invloss],
             {self.s1: s1, self.s2: s2, self.asample: asample})
        self.sumarize(error[0], step)
        return error[2:]

    def pred_act(self, s1, s2):
        '''
        returns action probability distribution predicted by inverse model
            input: s1,s2: [h, w, ch]
            output: ainvprobs: [ac_space]
        '''
        #sess = tf.get_default_session()
        return self.sess.run(self.logits, {self.s1: s1, self.s2: s2})[0, :]

    def pred_bonus(self, s1, s2, asample):
        '''
        returns bonus predicted by forward model
            input: s1,s2: [h, w, ch], asample: [ac_space] 1-hot encoding
            output: scalar bonus
        '''
        #sess = tf.get_default_session()
        error = self.sess.run([self.forwardloss, self.invloss],
             {self.s1: s1, self.s2: s2, self.asample: asample})
        print('ErrorF: ', error[0], ' ErrorI:', error[1])
        #error = self.sess.run(self.forwardloss,
        #    {self.s1: [s1], self.s2: [s2], self.asample: [asample]})
        #error = error * 0.01
        return error

class ReinforcementLearner(object):
    def __init__(self, input_shape, action_len):
        self.input = tf.placeholder(tf.float32, input_shape)
        l1 = tf.layers.conv2d(self.input, filters=32, kernel_size=[3,3], padding="same", activation=tf.nn.relu, name="cnn1")
        mp1 = tf.layers.max_pooling2d(l1, pool_size=[2,2],strides=1)
        l2 = tf.layers.conv2d(mp1, filters=64, kernel_size=[3,3], padding="same", activation=tf.nn.relu, name="cnn2")
        mp2 = tf.layers.max_pooling2d(l2, pool_size=[2,2],strides=1)
        l3 = tf.layers.conv2d(mp2, filters=64, kernel_size=[3,3], padding="same", activation=tf.nn.relu, name="cnn3")
        mp3 = tf.layers.max_pooling2d(l3, pool_size=[2,2],strides=1)
        f = tf.layers.flatten(mp3)
        self.Qout = tf.layers.dense(f, action_len, activation=tf.sigmoid)

        self.pred_act = tf.argmax(self.Qout, 1)
        self.nextQ = tf.placeholder(shape=[None, action_len], dtype=tf.float32)
        loss = tf.reduce_sum(tf.square(self.nextQ - self.Qout))
        self.trainer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
        self.updateModel = self.trainer.minimize(loss)

    def set_session(self, sess):
        self.sess = sess

    def get_action(self, s):
        a = self.sess.run(self.pred_act, feed_dict={self.input: s})
        return a

    def get_Qs(self, s):
        return self.sess.run(self.Qout, feed_dict={self.input: s})

    def train(self, s, sQ, s1Q, a, r, e):
        targetQ = sQ
        maxQ1 = np.max(s1Q)
        targetQ[0][a] = e * r + (1-e) * maxQ1
        self.sess.run(self.updateModel, feed_dict={self.input: s, self.nextQ: targetQ})