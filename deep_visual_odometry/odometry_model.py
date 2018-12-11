import tensorflow as tf
import numpy as np
import time

class OdomModel(object):
    def __init__(self, num_classes, batch_size=64, num_steps=50, cell_type='LSTM',
                 rnn_size=128, num_layers=2, learning_rate=0.001,
                 grad_clip=5, train_keep_prob=0.5, sampling=False):
        '''
        Initialize the input parameter to define the network
        inputs:
        :param num_classes: (int) the vocabulary size of your input data
        :param batch_size: (int) number of sequences in one batch
        :param num_steps: (int) length of each seuqence in one batch
        :param cell_type: your rnn cell type, 'LSTM' or 'GRU'
        :param rnn_size: (int) number of units in one rnn layer
        :param num_layers: (int) number of rnn layers
        :param learning_rate: (float)
        :param grad_clip: constraint of gradient to avoid gradient explosion
        :param train_keep_prob: (float) dropout probability for rnn cell training
        :param sampling: (boolean) whether train mode or sample mode
        '''
        # if not training
        if sampling == True:
            batch_size, num_steps = 1, 1
        else:
            batch_size, num_steps = batch_size, num_steps

        tf.reset_default_graph()

        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.cell_type = cell_type
        self.rnn_size = rnn_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.grad_clip = grad_clip
        self.train_keep_prob = train_keep_prob

        self.inputs_layer()
        self.rnn_layer()
        self.outputs_layer()
        self.loss()
        self.optimizer()
        self.saver = tf.train.Saver()

    def inputs_layer(self):
        '''
        build the input layer
        '''
        self.inputs = tf.placeholder(tf.int32, shape=(self.batch_size, self.num_steps, 2), name='inputs')
        self.targets = tf.placeholder(tf.int32, shape=(self.batch_size, self.num_steps, 3), name='targets')

        # add keep_prob
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        # one_hot encoding

    def rnn_layer(self):
        '''
        build rnn_cell layer
        we will use the paramters:
        self.cell_type, self.rnn_size, self.keep_prob, self.num_layers,
        self.batch_size, self.rnn_inputs
        we have to define:
        self.rnn_outputs, self.final_state for later use
        '''

        def build_cell():
            if self.cell_type == 'LSTM':
                cell = tf.nn.rnn_cell.LSTMCell(self.rnn_size, state_is_tuple=True)
            elif self.cell_type == 'GRU':
                cell = tf.nn.rnn_cell.GRUCell(self.rnn_size)
            else:
                raise ValueError('unkown cell type')
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.keep_prob)
            return cell

        rnn_layers = [build_cell() for i in range(self.num_layers)]
        multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers, state_is_tuple=True)
        self.initial_state = multi_rnn_cell.zero_state(self.batch_size, dtype=tf.float32)
        self.rnn_outputs, self.final_state = tf.nn.dynamic_rnn(multi_rnn_cell, inputs=self.inputs, dtype=tf.float32,
                                                               initial_state=self.initial_state)

    def outputs_layer(self):
        '''
        build the output layer
        '''
        # concate the output of rnn_cell，example: [[1,2,3],[4,5,6]] -> [1,2,3,4,5,6]
        seq_output = tf.concat(self.rnn_outputs, axis=1)  # tf.concat(concat_dim, values)
        # reshape
        x = tf.reshape(seq_output, [-1, self.rnn_size])

        # define mse layer variables:
        with tf.variable_scope('mse'):
            mse_w = tf.Variable(tf.truncated_normal([self.rnn_size, 3], stddev=0.1))
            mse_b = tf.Variable(tf.zeros(3))

        # calculate logits
        self.logits = tf.matmul(x, mse_w) + mse_b

        # softmax generate probability predictions
        # self.prob_pred = tf.nn.softmax(self.logits, name='predictions')

    def loss(self):
        '''
        calculate loss according to logits and targets
        '''

        # MSE loss
        loss = tf.squared_difference(self.logits, self.targets)
        self.loss = tf.reduce_mean(loss)

    def optimizer(self):
        '''
        build our optimizer
        Unlike previous worries of gradient vanishing problem,
        for some structures of rnn cells, the calculation of hidden layers' weights
        may lead to an "exploding gradient" effect where the value keeps growing.
        To mitigate this, we use the gradient clipping trick. Whenever the gradients are updated,
        they are "clipped" to some reasonable range (like -5 to 5) so they will never get out of this range.
        parameters we will use:
        self.loss, self.grad_clip, self.learning_rate
        we have to define:
        self.optimizer for later use
        '''
        # using clipping gradients
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        # optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
        grads = optimizer.compute_gradients(self.loss)
        clipped_grads = [(tf.clip_by_value(grad, -self.grad_clip, self.grad_clip), var) for grad, var in grads]
        self.optimizer = optimizer.apply_gradients(clipped_grads)

    def train(self, batches, max_count, save_every_n):
        self.session = tf.Session()

        with self.session as sess:
            sess.run(tf.global_variables_initializer())
            counter = 0
            new_state = sess.run(self.initial_state)
            # Train network
            for x, y in batches:
                counter += 1
                start = time.time()
                feed = {self.inputs: x,
                        self.targets: y,
                        self.keep_prob: self.train_keep_prob,
                        self.initial_state: new_state}
                batch_loss, new_state, _ = sess.run([self.loss,
                                                     self.final_state,
                                                     self.optimizer],
                                                    feed_dict=feed)

                end = time.time()
                if counter % 200 == 0:
                    print('step: {} '.format(counter),
                          'loss: {:.4f} '.format(batch_loss),
                          '{:.4f} sec/batch'.format((end - start)))

                if (counter % save_every_n == 0):
                    self.saver.save(sess, "checkpoints/i{}_l{}.ckpt".format(counter, self.rnn_size))

                if counter >= max_count:
                    break

            self.saver.save(sess, "checkpoints/i{}_l{}.ckpt".format(counter, self.rnn_size))

    #TODO: Replace with test()
    
    # cite: https://github.com/Oceanland-428/Pedestrian-Trajectories-Prediction-with-RNN/blob/master/train_test_LSTM.py
    
    def test(self, checkpoint, testing_X, batch_size):
        with tf.Session() as sess:
            self.saver.restore(sess, checkpoint)
            initial_state = sess.run(self.initial_state)
            test_prediction = np.empty([int(len(testing_X)/batch_size)*batch_size, 3])
            for batch in range(int(len(testing_X)/batch_size)):
                x_batch = testing_X[batch*batch_size:(batch+1)*batch_size,:]
                pre,initial_state = sess.run([self.logits,self.initial_state], feed_dict={self.inputs: x_batch,
                                                       self.keep_prob: 1,
                                                       self.initial_state: initial_state})
                test_prediction[batch*batch_size : (batch+1)*batch_size, :] = pre
        
        return test_prediction
    
    
    def sample(self, checkpoint, n_samples, vocab_size, vocab_to_ind, ind_to_vocab, prime='You \n'):
        '''
        generate new text given the prime word
        inputs:
        :param n_samples: (int) number of characters you want to generate
        :param vocab_size: (int) number of vocabulary size of your input data
        :param vocab_to_ind, ind_to_vocab: mapping from unique characters to indices
        :param prime: (str) you new text starting word
        outputs:
        -a string of generated characters
        '''
        # change text into character list
        samples = [c for c in prime]
        prime_input = np.array([vocab_to_ind[c] for c in samples], dtype=np.int32)
        output = prime
        with tf.Session() as sess:
            self.saver.restore(sess, checkpoint)
            initial_state = sess.run(self.initial_state)
            # priming
            for x in prime_input:
                prob_pred, initial_state = sess.run([self.prob_pred, self.final_state],
                                                    feed_dict={self.inputs: x.reshape(1, 1),
                                                               self.keep_prob: 1.0,
                                                               self.initial_state: initial_state})  # # prediction
            for i in range(n_samples):
                ind = pick_top_n(prob_pred, vocab_size, top_n=3)
                output += ind_to_vocab[ind]
                prob_pred, initial_state = sess.run([self.prob_pred, self.final_state],
                                                    feed_dict={self.inputs: ind.reshape(1, 1), self.keep_prob: 1.0,
                                                               self.initial_state: initial_state})
        return output


