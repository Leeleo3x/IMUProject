import numpy as np
import tensorflow as tf
from scipy.ndimage.filters import gaussian_filter1d
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

import sys
import os

sys.path.append('/home/yanhang/Documents/research/IMUProject/code/python')
from speed_regression import training_data as td

args = None


def get_batch(input_feature, input_target, batch_size, num_steps):
    total_num, dim = input_feature.shape
    assert input_target.shape[0] == total_num

    partition_length = total_num // batch_size
    feature_batches = np.empty([batch_size, partition_length, dim])
    target_batches = np.empty([batch_size, partition_length, input_target.shape[1]])
    for i in range(batch_size):
        feature_batches[i] = input_feature[i * partition_length:(i+1) * partition_length, :]
        target_batches[i] = input_target[i * partition_length:(i+1) * partition_length, :]

    epoch_size = partition_length // num_steps

    for i in range(epoch_size):
        feat = feature_batches[:, i * num_steps: (i+1) * num_steps, :]
        targ = target_batches[:, i * num_steps: (i+1) * num_steps, :]
        yield (feat, targ)


def construct_graph(input_dim, output_dim):
    # construct graph
    # placeholders for input and output
    x = tf.placeholder(tf.float32, [None, None, input_dim],
                       name='input_placeholder')
    y = tf.placeholder(tf.float32, [None, None, output_dim],
                       name='output_placeholder')
    init_state = tf.placeholder(tf.float32, [args.num_layer, 2, None, args.state_size])
    cell = tf.contrib.rnn.BasicLSTMCell(args.state_size, state_is_tuple=True)
    multi_cell = tf.contrib.rnn.MultiRNNCell([cell] * args.num_layer, state_is_tuple=True)

    rnn_outputs, final_state = tf.nn.dynamic_rnn(multi_cell, x, initial_state=tuple([tf.contrib.rnn.LSTMStateTuple(
        init_state[i][0], init_state[i][1]) for i in range(args.num_layer)]))

    # Fully connected layer
    init_stddev = 0.01
    fully_dims = [512, 256, 128]
    with tf.variable_scope('fully_connected1'):
        W1 = tf.get_variable('W1', shape=[args.state_size, fully_dims[0]],
                             initializer=tf.random_normal_initializer(stddev=init_stddev))
        b1 = tf.get_variable('b1', shape=[fully_dims[0]],
                             initializer=tf.random_normal_initializer(stddev=init_stddev))
        out_fully1 = tf.tanh(tf.matmul(tf.reshape(rnn_outputs, [-1, args.state_size]), W1) + b1)
    with tf.variable_scope('fully_connected2'):
        W2 = tf.get_variable('W2', shape=[fully_dims[0], fully_dims[1]],
                             initializer=tf.random_normal_initializer(stddev=init_stddev))
        b2 = tf.get_variable('b2', shape=[fully_dims[1]])
        out_fully2 = tf.tanh(tf.matmul(out_fully1, W2) + b2)
    with tf.variable_scope('fully_connected3'):
        W3 = tf.get_variable('W3', shape=[fully_dims[1], fully_dims[2]],
                             initializer=tf.random_normal_initializer(stddev=init_stddev))
        b3 = tf.get_variable('b3', shape=[fully_dims[fully_dims[2]]],
                             initializer=tf.random_normal_initializer(stddev=init_stddev))
        out_fully3 = tf.tanh(tf.matmul(out_fully2, W3), b3)

    # output layer
    with tf.variable_scope('output_layer'):
        W = tf.get_variable('W', shape=[fully_dims[-1], output_dim])
        b = tf.get_variable('b', shape=[output_dim], initializer=tf.random_normal_initializer(stddev=init_stddev))
    # regressed = tf.matmul(tf.reshape(rnn_outputs, [-1, args.state_size]), W) + b
    regressed = tf.matmul(out_fully3, W) + b
    return x, y, init_state, final_state, regressed


def run_training(features, targets, num_epoch, verbose=True, output_path=None,
                 tensorboard_path=None, checkpoint_path=None):
    assert len(features) == len(targets)
    assert features[0].ndim == 2

    input_dim = features[0].shape[1]
    output_dim = targets[0].shape[1]

    tf.reset_default_graph()
    # construct graph
    x, y, init_state, final_state, regressed = construct_graph(input_dim, output_dim)
    # loss and training step
    total_loss = tf.reduce_mean(tf.squared_difference(tf.reshape(regressed, [-1, output_dim]),
                                                      tf.reshape(y, [-1, output_dim])))
    tf.summary.scalar('mean squared loss', total_loss)
    tf.add_to_collection('total_loss', total_loss)
    tf.add_to_collection('rnn_input', x)
    tf.add_to_collection('rnn_output', y)
    tf.add_to_collection('init_state', init_state)
    tf.add_to_collection('regressed', regressed)
    tf.add_to_collection('state_size', args.state_size)
    tf.add_to_collection('num_layer', args.num_layer)

    global_step = tf.Variable(0, name='global_step', trainable=False)
    learning_rate = tf.train.exponential_decay(args.learning_rate, global_step, args.decay_step, args.decay_rate)

    # train_step = tf.train.AdagradOptimizer(learning_rate).minimize(total_loss, global_step=global_step)
    train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(total_loss, global_step=global_step)
    all_summary = tf.summary.merge_all()

    report_interval = 100
    global_counter = 0
    with tf.Session() as sess:
        train_writer = None
        if tensorboard_path is not None:
            train_writer = tf.summary.FileWriter(tensorboard_path, sess.graph)

        sess.run(tf.global_variables_initializer())
        training_losses = []
        saver = None
        if output_path is not None:
            saver = tf.train.Saver()
        for i in range(num_epoch):
            if verbose:
                print('EPOCH', i)
            epoch_loss = 0.0
            steps_in_epoch = 0
            for data_id in range(len(features)):
                state = tuple(
                    [(np.zeros([args.batch_size, args.state_size]), np.zeros([args.batch_size, args.state_size]))
                     for i in range(args.num_layer)])
                for _, (X, Y) in enumerate(get_batch(features[data_id], targets[data_id],
                                                     args.batch_size, args.num_steps)):
                    summaries, current_loss, state, _ = sess.run([all_summary,
                                                                  total_loss,
                                                                  final_state,
                                                                  train_step], feed_dict={x: X, y: Y, init_state: state})
                    epoch_loss += current_loss
                    # if tensorboard_path is not None:
                    #     train_writer.add_summary(summaries, global_counter)
                    if (checkpoint_path is not None) and global_counter % args.checkpoint == 0 and global_counter > 0:
                        saver.save(sess, checkpoint_path + '/ckpt', global_step=global_counter)
                        print('Checkpoint file saved at step', global_counter)
                    # if global_step % report_interval == 0 and global_step > 0 and verbose:
                    #     pass
                    steps_in_epoch += 1
                    global_counter += 1
            training_losses.append(epoch_loss / steps_in_epoch)
            print('Average loss at epoch {:d} (step {:d}): {:f}'.format(i, global_counter, epoch_loss / steps_in_epoch))
        if output_path is not None:
            saver.save(sess, output_path)
        print('Meta graph saved to', output_path)

        # testing
        # state = tuple([(np.zeros([args.batch_size, args.state_size]), np.zeros([args.batch_size, args.state_size]))
        #                for i in range(args.num_layer)])

        # test_result_whole, test_loss_whole = sess.run([regressed, total_loss],
        #                                               feed_dict={x: features[0].reshape([1, -1, 6]),
        #                                                          y: targets[0].reshape([1, -1, 1]),
        #                                                          init_state: state})
        # print('test loss whole:', mean_squared_error(test_result_whole.reshape([-1, 1]), targets[0].reshape([-1, 1])))
        
        # plt.figure('test')
        # plt.plot(test_result_whole.reshape([-1, 1]))
        # plt.plot(targets[0].reshape([-1, 1]))
        # plt.legend(['predicted', 'gt'])
        # plt.show()
 
    return training_losses


if __name__ == '__main__':
    import pandas
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('list', type=str)
    parser.add_argument('--feature_smooth_sigma', type=float, default=-1)
    parser.add_argument('--target_smooth_sigma', type=float, default=30.0)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_steps', type=int, default=500)
    parser.add_argument('--state_size', type=int, default=500)
    parser.add_argument('--num_layer', type=int, default=3)
    parser.add_argument('--num_epoch', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--decay_step', type=int, default=1000)
    parser.add_argument('--decay_rate', type=float, default=0.95)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--checkpoint', type=int, default=5000)
    args = parser.parse_args()

    root_dir = os.path.dirname(args.list)
    imu_columns = ['gyro_stab_x', 'gyro_stab_y', 'gyro_stab_z',
                   'linacce_stab_x', 'linacce_stab_y', 'linacce_stab_z']

    with open(args.list) as f:
        datasets = f.readlines()

    nano_to_sec = 1e09
    features_all = []
    targets_all = []
    total_samples = 0
    for data in datasets:
        data_name = data.strip()
        data_all = pandas.read_csv(root_dir + '/' + data_name + '/processed/data.csv')
        ts = data_all['time'].values / nano_to_sec
        gravity = data_all[['grav_x', 'grav_y', 'grav_z']].values
        position = data_all[['pos_x', 'pos_y', 'pos_z']].values
        orientation = data_all[['ori_w', 'ori_x', 'ori_y', 'ori_z']].values
        print('Loading ' + data_name + ', samples:', ts.shape[0])

        feature_vectors = data_all[imu_columns].values
        if args.feature_smooth_sigma > 0:
            feature_vectors = gaussian_filter1d(feature_vectors, sigma=args.feature_smooth_sigma, axis=0)
        # get training data

        target_speed = td.compute_local_speed_with_gravity(ts, position, orientation, gravity)
        if args.target_smooth_sigma > 0:
            target_speed = gaussian_filter1d(target_speed, sigma=args.target_smooth_sigma, axis=0)
        features_all.append(feature_vectors.astype(np.float32))
        targets_all.append(target_speed[:, 2:3].astype(np.float32))
        total_samples += target_speed.shape[0]

    # configure output path
    output_root = None
    if args.output is None:
        output_root = '../../../models/LSTM'
    else:
        output_root = args.output
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    model_path = output_root + '/model.tf'
    tfboard_path = output_root + '/tensorboard'
    chpt_path = output_root + '/checkpoints/'
    if not os.path.exists(tfboard_path):
        os.makedirs(tfboard_path)
    if not os.path.exists(chpt_path):
        os.makedirs(chpt_path)

    print('Total number of samples: ', total_samples)
    print('Running training')
    losses = run_training(features_all, targets_all, args.num_epoch,
                          output_path=model_path, tensorboard_path=tfboard_path, checkpoint_path=chpt_path)
    # plt.plot(losses)
    # plt.show()
