import tensorflow as tf
import numpy as np
import time
import deep_visual_odometry.kitti_utils as kitti

class OdomModelV2(object):
    def __init__(self, batch_size=64, num_steps=50, cell_type='LSTM',
                 rnn_size=128, num_layers=2, learning_rate=0.001,
                 grad_clip=5, lamda_l2_reg = 0.001,train_keep_prob=0.5, sampling=False):
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

        #self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.cell_type = cell_type
        self.rnn_size = rnn_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.grad_clip = grad_clip
        self.train_keep_prob = train_keep_prob
        self.lamda_l2_reg = lamda_l2_reg

        self.inputs_layer()
        self.rnn_layer()
        self.loss()
        self.optimizer()
        self.saver = tf.train.Saver()
        print('odometry model initialized')

    def inputs_layer(self):
        '''
        build the input layer
        '''
        self.inputs = tf.placeholder(tf.float32, shape=(self.batch_size, self.num_steps, 2), name='inputs')
        self.targets = tf.placeholder(tf.float32, shape=(self.batch_size, self.num_steps, 3), name='targets')
        self.initial_poses = tf.placeholder(tf.float32, shape=(self.batch_size, 3), name='initial_poses')

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
                raise ValueError('unknown cell type')
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.keep_prob)
            return cell
        rnn_layers = [build_cell() for i in range(self.num_layers)]

        multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers, state_is_tuple=True)

        print('created multi layer rnn cell')

        # output affine projection projection parameters
        with tf.variable_scope('projection'):
            w = tf.Variable(tf.truncated_normal([self.rnn_size, 3], stddev=0.1))
            b = tf.Variable(tf.zeros(3))

        # getting initial zero state of multi_rnn_cell
        self.initial_state = multi_rnn_cell.zero_state(self.batch_size, dtype=tf.float32)
        state = self.initial_state

        # static rollout of rnn cells
        # differs from standard rollout as the output are also fed to into the next cell as input by concatenation
        outputs_ = []
        output_ = self.initial_poses
        for i in range(self.num_steps):
            input_ = tf.concat([self.inputs[:, i], output_], axis=1)
            output_, state = multi_rnn_cell(input_, state)
            # project to get pose output
            output_ = tf.tensordot(output_, w, axes=[[len(output_.shape) - 1], [0]]) + b
            outputs_.append(output_)
            # concatenate the output pose prediction with the input vel i.e [input, output]

        print('rolled out cell for num_steps ')

        # reshaping outputs to make it compatible with shape as in case of using dynamic_rnn
        outputs = []
        for i in range(self.batch_size):
            # output for a sequence
            output = tf.concat([tf.expand_dims(outputs_[j][i, :], 0) for j in range(self.num_steps)], axis=0)
            outputs.append(tf.expand_dims(output, axis=0))
        # batch of output sequences
        outputs = tf.concat(outputs, axis=0)

        print('reshaped outputs')

        self.outputs = outputs
        self.final_state = state

    def loss(self):
        '''
        calculate loss from outputs and targets
        '''

        print('shape of outputs',self.outputs.get_shape())
        print('shape of targets',self.targets.get_shape())
        # error = tf.squared_difference(out_RNN, self.targets)
        error = tf.losses.absolute_difference(self.outputs, self.targets)
        self.loss = tf.reduce_mean(error)

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
        l2 = self.lamda_l2_reg * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables() 
                                            if not ("noreg" in tf_var.name or "Bias" in tf_var.name))
        self.loss += l2
        grads = optimizer.compute_gradients(self.loss)
        clipped_grads = [(tf.clip_by_value(grad, -self.grad_clip, self.grad_clip), var) for grad, var in grads]
        self.optimizer = optimizer.apply_gradients(clipped_grads)
    
    # cite: https://github.com/Oceanland-428/Pedestrian-Trajectories-Prediction-with-RNN/blob/master/train_test_LSTM.py

    def train(self, kitti_data_obj, max_count, save_every_n, sequences):
        self.session = tf.Session()
        with self.session as sess:
            merge = tf.summary.merge_all()
            writer = tf.summary.FileWriter("log/{}".format('RNN_velocities_to_pose_model'), self.session.graph)
            sess.run(tf.global_variables_initializer())
            counter = 0
            new_state = sess.run(self.initial_state)

            # Train network
            print('training .. ')
            loss = []
            while counter <= max_count:
                counter += 1
                start = time.time()
                _, velocities_batch, poses_batch = kitti_data_obj.get_series_batch_train(batch_size = self.batch_size,
                                                                                         sequences = sequences)
                inputs_velocities = velocities_batch[:, :, 0:2]
                initial_poses = velocities_batch[:, 0, 2:5]
                feed = {self.inputs: inputs_velocities,
                        self.targets: poses_batch,
                        self.keep_prob: self.train_keep_prob,
                        self.initial_state: new_state,
                        self.initial_poses: initial_poses}

                batch_loss, _  = sess.run([self.loss, self.optimizer], feed_dict =feed)
                loss.append(batch_loss)

                end = time.time()
                if counter % 1 == 0:
                    print('step: {} '.format(counter),
                          'loss: {:.4f} '.format(batch_loss),
                          '{:.4f} sec/batch'.format((end - start)))


                if (counter % save_every_n == 0):
                    self.saver.save(sess, "checkpoints/i{}_l{}.ckpt".format(counter, self.rnn_size))

            self.saver.save(sess, "checkpoints/i{}_l{}.ckpt".format(counter, self.rnn_size))
        return loss
    
    def test_batch(self, checkpoint, testing_X, batch_size):
        with tf.Session() as sess:
            self.saver.restore(sess, checkpoint)
            initial_state = sess.run(self.initial_state)
            test_prediction = np.empty([int(len(testing_X)/batch_size)*batch_size, self.num_steps, 3])
            for batch in range(int(len(testing_X)/batch_size)):
                x_batch = testing_X[batch*batch_size:(batch+1)*batch_size,:]
                pre , initial_state = sess.run([self.outputs,self.initial_state], feed_dict={self.inputs: x_batch,
                                                       self.keep_prob: 1,
                                                       self.initial_state: initial_state})
                test_prediction[batch*batch_size : (batch+1)*batch_size, :] = pre
        
        return test_prediction

    def test(self, checkpoint, X, initial_pose):
        """

        X : (-1, 2)
        initial_pose = (3,)

        """

        num_samples = X.shape[0]

        with tf.Session() as sess:
            self.saver.restore(sess, checkpoint)
            initial_state = sess.run(self.initial_state)
            y_pred = np.zeros([num_samples, 3])
            y_pred[0] = initial_pose
            for i in range(num_samples):
                x_new = X[i]
                feed = {self.inputs: np.array([[x_new]]),
                        self.keep_prob: 1,
                        self.initial_state: initial_state,
                        self.initial_poses: np.array([y_pred[i]])}
                y_pred[i], initial_state = sess.run([self.outputs, self.final_state], feed_dict = feed)
        return y_pred





    

