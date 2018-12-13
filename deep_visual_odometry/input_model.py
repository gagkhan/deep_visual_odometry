import tensorflow as tf
import numpy as np
import deep_visual_odometry.cnn as CNN

def InputCNN(input_x, input_y, is_training,
          img_len=32, channel_num=6, output_size=3,
          conv_featmap=[16,16,16,16], fc_units=[128,128],
          conv_kernel_size=[7,5,5,5], pooling_size=[2,2,2,2],
          l2_norm=0.01, seed=235):
    assert len(conv_featmap) == len(conv_kernel_size) and len(conv_featmap) == len(pooling_size)

    # conv layer
    conv_layer_0 = CNN.conv_layer(input_x=input_x,
                              in_channel=channel_num,
                              out_channel=conv_featmap[0],
                              kernel_shape=conv_kernel_size[0],
                              rand_seed=seed,
                              index=0)
    # pooling layer
    pooling_layer_0 = CNN.max_pooling_layer(input_x=conv_layer_0.output(),
                                        k_size=pooling_size[0],
                                        padding="VALID")

    # flatten
    conv_layer_1 = CNN.conv_layer(input_x=pooling_layer_0.output(),
                              in_channel=conv_featmap[0],
                              out_channel=conv_featmap[1],
                              kernel_shape=conv_kernel_size[1],
                              rand_seed=seed,
                              index=1,
                              y_stride=2)
    
    pooling_layer_1 = CNN.max_pooling_layer(input_x=conv_layer_1.output(),
                                        k_size=pooling_size[1],
                                        padding="VALID")
    
    conv_layer_2 = CNN.conv_layer(input_x=pooling_layer_1.output(),
                              in_channel=conv_featmap[1],
                              out_channel=conv_featmap[2],
                              kernel_shape=conv_kernel_size[2],
                              rand_seed=seed,
                              index=2,
                              y_stride=2)
    
    pooling_layer_2 = CNN.max_pooling_layer(input_x=conv_layer_2.output(),
                                        k_size=pooling_size[2],
                                        padding="VALID")
    
    conv_layer_3 = CNN.conv_layer(input_x=pooling_layer_2.output(),
                              in_channel=conv_featmap[2],
                              out_channel=conv_featmap[3],
                              kernel_shape=conv_kernel_size[3],
                              rand_seed=seed,
                              index=3,
                              x_stride=2,
                              y_stride=2)
    
    pooling_layer_3 = CNN.max_pooling_layer(input_x=conv_layer_3.output(),
                                        k_size=pooling_size[3],
                                        padding="VALID")
    
    pool_shape = pooling_layer_3.output().get_shape()
    img_vector_length = pool_shape[1].value * pool_shape[2].value * pool_shape[3].value
    flatten = tf.reshape(pooling_layer_3.output(), shape=[-1, img_vector_length])
    
    # fc layer
    fc_layer_0 = CNN.fc_layer(input_x=flatten,
                          in_size=img_vector_length,
                          out_size=fc_units[0],
                          rand_seed=seed,
                          keep_prob=.9,
                          activation_function=tf.nn.relu,
                          index=0)
    
    #fc layer
    fc_layer_1 = CNN.fc_layer(input_x=fc_layer_0.output(),
                          in_size=fc_units[0],
                          out_size=fc_units[1],
                          rand_seed=seed,
                          keep_prob=.9,
                          activation_function=tf.nn.relu,
                          index=1)

    fc_layer_2 = CNN.fc_layer(input_x=fc_layer_1.output(),
                          in_size=fc_units[1],
                          out_size=output_size,
                          rand_seed=seed,
                          keep_prob=.9,
                          activation_function=tf.nn.relu,
                          index=2)

    # saving the parameters for l2_norm loss
    conv_w = [conv_layer_0.weight,conv_layer_1.weight,conv_layer_2.weight,conv_layer_3.weight]
    fc_w = [fc_layer_0.weight, fc_layer_1.weight,fc_layer_2.weight]

    # loss
    with tf.name_scope("loss"):
        l2_loss = tf.reduce_sum([tf.norm(w) for w in fc_w])
        l2_loss += tf.reduce_sum([tf.reduce_sum(tf.norm(w, axis=[-2, -1])) for w in conv_w])

        mse_loss = tf.reduce_mean(
            tf.squared_difference(fc_layer_2.output(),input_y),
            name='mse')
        loss = tf.add(mse_loss, l2_norm * l2_loss, name='loss')

        tf.summary.scalar('Loss', loss)

    return fc_layer_2.output(), loss


def mse(output, input_y):
    with tf.name_scope('mse'):
        ce = tf.reduce_mean(tf.squared_difference(output,input_y))

    return ce


def train_step(loss, learning_rate=1e-3):
    with tf.name_scope('train_step'):
        step = tf.train.AdamOptimizer().minimize(loss)

    return step

"""
def evaluate(output, input_y):
    with tf.name_scope('evaluate'):
        pred = tf.argmax(output, axis=1)
        error_num = tf.count_nonzero(pred - input_y, name='error_num')
        tf.summary.scalar('Error_num', error_num)
    return error_num
    """

def training(X_train, y_train, X_val, y_val,
             conv_featmap=[16,16,16,16],
             fc_units=[128,128],
             conv_kernel_size=[7,5,5,5],
             pooling_size=[2,2,2,2],
             l2_norm=0.01,
             seed=235,
             learning_rate=1e-2,
             epoch=20,
             batch_size=245,
             verbose=False,
             pre_trained_model=None):

    print("Building velocity CNN. Parameters: ")
    print("conv_featmap={}".format(conv_featmap))
    print("fc_units={}".format(fc_units))
    print("conv_kernel_size={}".format(conv_kernel_size))
    print("pooling_size={}".format(pooling_size))
    print("l2_norm={}".format(l2_norm))
    print("seed={}".format(seed))
    print("learning_rate={}".format(learning_rate))

    # define the variables and parameter needed during training
    xs = tf.placeholder(shape=[None, 50, 150, 6], dtype=tf.float32,name="xs")
    ys = tf.placeholder(shape=[None, 2], dtype=tf.float32,name="ys")
    is_training = tf.placeholder(tf.bool, name="is_training")

    output, loss = InputCNN(xs, ys, is_training,
                         img_len=32,
                         channel_num=6,
                         output_size=2,
                         conv_featmap=conv_featmap,
                         fc_units=fc_units,
                         conv_kernel_size=conv_kernel_size,
                         pooling_size=pooling_size,
                         l2_norm=l2_norm,
                         seed=seed)

    iters = int(X_train.shape[0] / batch_size)
    print('number of batches for training: {}'.format(iters))

    step = train_step(loss)
    mserr = mse(output, ys)

    iter_total = 0
    best_mse = np.inf
    cur_model_name = 'CNN_Velocity_Model'

    with tf.Session() as sess:
        merge = tf.summary.merge_all()

        writer = tf.summary.FileWriter("log/{}".format(cur_model_name), sess.graph)
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        # try to restore the pre_trained
        if pre_trained_model is not None:
            try:
                print("Load the model from: {}".format(pre_trained_model))
                saver.restore(sess, 'model/{}'.format(pre_trained_model))
            except Exception:
                raise ValueError("Load model Failed!")

        for epc in range(epoch):
            print("epoch {} ".format(epc + 1))

            for itr in range(iters):
                iter_total += 1

                training_batch_x = X_train[itr * batch_size: (1 + itr) * batch_size]
                training_batch_y = y_train[itr * batch_size: (1 + itr) * batch_size]

                _, cur_loss = sess.run([step, loss], feed_dict={xs: training_batch_x,
                                                                ys: training_batch_y,
                                                                is_training: True})

                if iter_total % 100 == 0:
                    # do validation
                    valid_mse, merge_result = sess.run([mserr, merge], feed_dict={xs: X_val,
                                                                                ys: y_val,
                                                                                is_training: False})
                    
                    if verbose:
                        print('{}/{} loss: {} validation mse : {}%'.format(
                            batch_size * (itr + 1),
                            X_train.shape[0],
                            cur_loss,
                            valid_mse))

                    # save the merge result summary
                    writer.add_summary(merge_result, iter_total)

                    # when achieve the best validation accuracy, we store the model paramters
                    if valid_mse < best_mse:
                        print('Best validation mse! iteration:{} val_mse: {}'.format(iter_total, valid_mse))
                        best_mse = valid_mse
                        saver.save(sess, 'model/{}'.format(cur_model_name))

    print("Traning ends. The best valid mse is {}. Model named {}.".format(best_mse, cur_model_name))


def test_input_model(pre_trained_model,X_test, y_test):

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('model/CNN_Velocity_Model.meta')
        saver.restore(sess,tf.train.latest_checkpoint('model/'))
        graph = tf.get_default_graph()
        xs = graph.get_tensor_by_name("xs:0")
        ys = graph.get_tensor_by_name("ys:0")
        is_training = graph.get_tensor_by_name("is_training:0")
        output = graph.get_tensor_by_name("fc_layer_2/output_2:0")
            
        test_out = sess.run(output, feed_dict={xs: X_test,
                                               ys: y_test,
                                               is_training: False})
    
    return test_out
        
        
        
    
