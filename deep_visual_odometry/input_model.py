import tensorflow as tf
import numpy as np
import cnn as CNN

def Input_CNN(input_x, input_y, is_training,
          img_len=32, channel_num=3, output_size=3,
          conv_featmap=[16], fc_units=[84],
          conv_kernel_size=[5, 5], pooling_size=[2, 2],
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
                              x_stride=2)
    
    pooling_layer_1 = CNN.max_pooling_layer(input_x=conv_layer_1.output(),
                                        k_size=pooling_size[1],
                                        padding="VALID")
    
    conv_layer_2 = CNN.conv_layer(input_x=pooling_layer_1.output(),
                              in_channel=conv_featmap[1],
                              out_channel=conv_featmap[2],
                              kernel_shape=conv_kernel_size[2],
                              rand_seed=seed,
                              index=2,
                              x_stride=2)
    
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
                          keep_prob=.75,
                          activation_function=tf.nn.relu,
                          index=2)

    # saving the parameters for l2_norm loss
    conv_w = [conv_layer_0.weight,conv_layer_1.weight,conv_layer_2.weight,conv_layer_3.weight]
    fc_w = [fc_layer_0.weight, fc_layer_1.weight,fc_layer_2.weight]

    # loss
    with tf.name_scope("loss"):
        l2_loss = tf.reduce_sum([tf.norm(w) for w in fc_w])
        l2_loss += tf.reduce_sum([tf.reduce_sum(tf.norm(w, axis=[-2, -1])) for w in conv_w])

        label = tf.one_hot(input_y, 10)
        cross_entropy_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=fc_layer_2.output()),
            name='cross_entropy')
        loss = tf.add(cross_entropy_loss, l2_norm * l2_loss, name='loss')

        tf.summary.scalar('LeNet_loss', loss)

    return fc_layer_2.output(), loss


def cross_entropy(output, input_y):
    with tf.name_scope('cross_entropy'):
        label = tf.one_hot(input_y, 10)
        ce = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=output))

    return ce


def train_step(loss, learning_rate=1e-3):
    with tf.name_scope('train_step'):
        step = tf.train.AdamOptimizer().minimize(loss)

    return step


def evaluate(output, input_y):
    with tf.name_scope('evaluate'):
        pred = tf.argmax(output, axis=1)
        error_num = tf.count_nonzero(pred - input_y, name='error_num')
        tf.summary.scalar('LeNet_error_num', error_num)
    return error_num