import numpy as np
import tensorflow as tf
from scipy.ndimage.filters import gaussian_filter1d
from sklearn.metrics import r2_score, mean_squared_error
import sys
import os
from speed_regression import training_data as td

args = None
nano_to_sec = 1e09


def get_batch(input_feature, input_target, batch_size, num_steps, stride_ratio=1):
    total_num, dim = input_feature.shape
    assert input_target.shape[0] == total_num

    partition_length = total_num // batch_size
    feature_batches = np.empty([batch_size, partition_length, dim])
    target_batches = np.empty([batch_size, partition_length, input_target.shape[1]])
    for i in range(batch_size):
        feature_batches[i] = input_feature[i * partition_length:(i+1) * partition_length, :]
        target_batches[i] = input_target[i * partition_length:(i+1) * partition_length, :]

    stride = num_steps // stride_ratio
    epoch_size = partition_length // stride
    for i in range(epoch_size):
        if i * stride + num_steps >= feature_batches.shape[1]:
            break
        feat = feature_batches[:, i * stride: i * stride + num_steps, :]
        targ = target_batches[:, i * stride: i * stride + num_steps, :]
        yield (feat, targ)


def construct_graph(input_dim, output_dim, batch_size=1):
    # construct graph
    init_stddev = 0.001
    # fully_dims = [512, 256, 128]
    fully_dims = [512, 256]
    # placeholders for input and output
    x = tf.placeholder(tf.float32, [batch_size, None, input_dim],
                       name='input_placeholder')
    y = tf.placeholder(tf.float32, [batch_size, None, output_dim],
                       name='output_placeholder')

    cell = tf.contrib.rnn.BasicLSTMCell(args.state_size, state_is_tuple=True)
    multi_cell = tf.contrib.rnn.MultiRNNCell([cell] * args.num_layer, state_is_tuple=True)
    init_state = multi_cell.zero_state(batch_size, dtype=tf.float32)

    rnn_outputs, final_state = tf.nn.dynamic_rnn(multi_cell, x, initial_state=init_state)

    # Fully connected layer

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
    # with tf.variable_scope('fully_connected3'):
    #     W3 = tf.get_variable('W3', shape=[fully_dims[1], fully_dims[2]],
    #                          initializer=tf.random_normal_initializer(stddev=init_stddev))
    #     b3 = tf.get_variable('b3', shape=[fully_dims[2]],
    #                          initializer=tf.random_normal_initializer(stddev=init_stddev))
    #     out_fully3 = tf.tanh(tf.matmul(out_fully2, W3) + b3)

    # output layer
    with tf.variable_scope('output_layer'):
        W = tf.get_variable('W', shape=[fully_dims[-1], output_dim])
        b = tf.get_variable('b', shape=[output_dim], initializer=tf.random_normal_initializer(stddev=init_stddev))
    # regressed = tf.matmul(tf.reshape(rnn_outputs, [-1, args.state_size]), W) + b
    regressed = tf.matmul(out_fully2, W) + b
    return {'x': x, 'y': y, 'init_state': init_state, 'final_state': final_state, 'regressed': regressed}


def run_testing(sess, variable_dict, feature, target, init_state):
    input_dim, output_dim = feature.shape[1], target.shape[1]
    feature_rnn = feature.reshape([1, -1, input_dim])
    target_rnn = target.reshape([1, -1, output_dim])
    regressed_rnn, loss = sess.run([variable_dict['regressed'], variable_dict['total_loss']],
                             feed_dict={variable_dict['x']: feature_rnn, variable_dict['y']: target_rnn,
                                        variable_dict['init_state']: init_state})
    predicted = np.array(regressed_rnn).reshape([-1, output_dim])
    return predicted, loss


def run_training(features, targets, valid_features, valid_targets, num_epoch, verbose=True, output_path=None,
                 tensorboard_path=None, checkpoint_path=None):
    assert len(features) == len(targets)
    assert len(valid_features) == len(valid_targets)
    assert features[0].ndim == 2

    input_dim = features[0].shape[1]
    output_dim = targets[0].shape[1]

    # first compute the variable of all channels
    targets_concat = np.concatenate(targets, axis=0)

    target_mean = np.mean(targets_concat, axis=0)
    target_variance = np.var(targets_concat, axis=0)
    print('target mean:', target_mean)
    print('target variance:', target_variance)

    tf.add_to_collection('target_mean_x', target_mean[0])
    tf.add_to_collection('target_mean_z', target_mean[1])
    tf.add_to_collection('target_variance_x', target_variance[0])
    tf.add_to_collection('target_variance_z', target_variance[1])

    # normalize the input
    for i in range(len(targets)):
        targets[i] = np.divide(targets[i] - target_mean, target_variance)

    for i in range(len(valid_targets)):
        valid_targets[i] = np.divide(valid_targets[i] - target_mean, target_variance)
    valid_targets_concat = np.concatenate(valid_targets, axis=0)

    tf.reset_default_graph()
    # construct graph
    variable_dict = construct_graph(input_dim, output_dim, args.batch_size)
    x = variable_dict['x']
    y = variable_dict['y']
    regressed = variable_dict['regressed']
    init_state = variable_dict['init_state']
    final_state = variable_dict['final_state']

    # loss and training step
    total_loss = tf.reduce_mean(tf.squared_difference(tf.reshape(regressed, [-1, output_dim]), tf.reshape(y, [-1, output_dim])))
    variable_dict['total_loss'] = total_loss
    
    tf.summary.scalar('mean squared loss', total_loss)
    tf.add_to_collection('total_loss', total_loss)
    tf.add_to_collection('rnn_input', x)
    tf.add_to_collection('rnn_output', y)
    tf.add_to_collection('regressed', regressed)
    tf.add_to_collection('state_size', args.state_size)
    tf.add_to_collection('num_layer', args.num_layer)

    global_step = tf.Variable(0, name='global_step', trainable=False)
    learning_rate = tf.train.exponential_decay(args.learning_rate, global_step, args.decay_step, args.decay_rate)

    train_step = tf.train.AdagradOptimizer(learning_rate).minimize(total_loss, global_step=global_step)
    # train_step = tf.train.AdadeltaOptimizer(learning_rate).minimize(total_loss, global_step=global_step)
    # train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(total_loss, global_step=global_step)
    all_summary = tf.summary.merge_all()

    report_interval = 100
    global_counter = 0
    with tf.Session() as sess:
        train_writer = None
        if tensorboard_path is not None:
            train_writer = tf.summary.FileWriter(tensorboard_path, sess.graph)

        sess.run(tf.global_variables_initializer())
        training_losses = []
        validation_losses = []
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
                    if (checkpoint_path is not None) and global_counter % args.checkpoint == 0 and global_counter > 0:
                        saver.save(sess, checkpoint_path + '/ckpt', global_step=global_counter)
                        print('Checkpoint file saved at step', global_counter)
                    # if global_counter % report_interval == 0 and global_counter > 0 and verbose:
                    #     if tensorboard_path is not None:
                    #         train_writer.add_summary(summaries, global_counter)
                    steps_in_epoch += 1
                    global_counter += 1
            training_losses.append(epoch_loss / steps_in_epoch)
            print('Training loss at epoch {:d} (step {:d}): {:f}'.format(i, global_counter, training_losses[-1]))

            # run validation
            predicted_concat = []
            for valid_id in range(len(valid_features)):
                state = tuple(
                    [(np.zeros([1, args.state_size]), np.zeros([1, args.state_size]))
                     for i in range(args.num_layer)])
                predicted, cur_loss = run_testing(sess, variable_dict, valid_features[valid_id], valid_targets[valid_id], state)
                # loss_sklearn = mean_squared_error(np.reshape(np.array(predicted), [-1, 2]), valid_targets[valid_id])
                # print('Loss for valid set {}: {:.6f}(tf), {:.6f}(sklearn)'.format(valid_id, cur_loss, loss_sklearn))
                predicted_concat.append(predicted)
            predicted_concat = np.concatenate(predicted_concat, axis=0)
            l2_loss = np.array([mean_squared_error(predicted_concat[:, 0], valid_targets_concat[:, 0]),
                                mean_squared_error(predicted_concat[:, 1], valid_targets_concat[:, 1])])
            validation_losses.append(l2_loss)
            print('Validation loss at epoch {:d} (step {:d}):'.format(i, global_counter), validation_losses[-1],
                  np.average(l2_loss))

        if output_path is not None:
            saver.save(sess, output_path, global_step=global_counter)
            print('Meta graph saved to', output_path)

        # output final training loss
        # total_samples = 0
        # train_error_axis = np.zeros(output_dim, dtype=float)
        # for data_id in range(len(features)):
        #     feature_rnn = features[data_id].reshape([1, -1, input_dim])
        #     target_rnn = targets[data_id].reshape([1, -1, output_dim])
        #     predicted = sess.run([regressed], feed_dict={x: feature_rnn, y: target_rnn})
        #     diff = np.power(np.array(predicted).reshape([-1, output_dim]) - targets[data_id], 2)
        #     train_error_axis += np.sum(diff, axis=0)
        #     total_samples += features[data_id].shape[0]
        #print('Overall training loss:', train_error_axis / total_samples)

    return training_losses, validation_losses

def load_dataset(listpath, imu_columns, feature_smooth_sigma, target_smooth_sigma):
    root_dir = os.path.dirname(listpath)
    features_all = []
    targets_all = []
    with open(listpath) as f:
        datasets = f.readlines()
    for data in datasets:
        if data[0] == '#':
            continue
        [data_name, _] = [x.strip() for x in data.split(',')]
        data_all = pandas.read_csv(root_dir + '/' + data_name + '/processed/data.csv')
        ts = data_all['time'].values / nano_to_sec
        gravity = data_all[['grav_x', 'grav_y', 'grav_z']].values
        position = data_all[['pos_x', 'pos_y', 'pos_z']].values
        orientation = data_all[['ori_w', 'ori_x', 'ori_y', 'ori_z']].values
        print('Loading ' + data_name + ', samples:', ts.shape[0])

        feature_vectors = data_all[imu_columns].values
        if feature_smooth_sigma > 0:
            feature_vectors = gaussian_filter1d(feature_vectors, sigma=feature_smooth_sigma, axis=0)
        # get training data
        target_speed = td.compute_local_speed_with_gravity(ts, position, orientation, gravity)
        if target_smooth_sigma > 0:
            target_speed = gaussian_filter1d(target_speed, sigma=target_smooth_sigma, axis=0)
        features_all.append(feature_vectors)
        targets_all.append(target_speed[:, [0, 2]])
    return features_all, targets_all

if __name__ == '__main__':
    import pandas
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('list', type=str)
    parser.add_argument('validation', type=str)
    parser.add_argument('--feature_smooth_sigma', type=float, default=-1.0)
    parser.add_argument('--target_smooth_sigma', type=float, default=30.0)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_steps', type=int, default=400)
    parser.add_argument('--state_size', type=int, default=1500)
    parser.add_argument('--num_layer', type=int, default=1)
    parser.add_argument('--num_epoch', type=int, default=200)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--decay_step', type=int, default=1000)
    parser.add_argument('--decay_rate', type=float, default=0.95)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--checkpoint', type=int, default=5000)
    args = parser.parse_args()

    imu_columns = ['gyro_x', 'gyro_y', 'gyro_z',
                   'linacce_x', 'linacce_y', 'linacce_z',
                   'grav_x', 'grav_y', 'grav_z']

    print('---------------\nTraining set')
    features_train, targets_train = load_dataset(args.list, imu_columns,
                                                 args.feature_smooth_sigma, args.target_smooth_sigma)
    print('---------------\nValidation set')
    features_validation, targets_validation = load_dataset(args.validation, imu_columns,
                                                           args.feature_smooth_sigma, args.target_smooth_sigma)
    # configure output path
    output_root = None
    model_path = None
    tfboard_path = None
    chpt_path = None
    if args.output is not None:
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

    print('Total number of training samples: ', sum([len(target) for target in targets_train]))
    print('Total number of validation samples: ', sum([len(target) for target in targets_validation]))
    print('Running training')
    training_losses, validation_losses  = run_training(features_train, targets_train, features_validation, targets_validation, args.num_epoch,
                                                       output_path=model_path, tensorboard_path=tfboard_path, checkpoint_path=chpt_path)

    if output_root is not None:
        assert len(training_losses) == len(validation_losses)
        with open(output_root + '/losses.txt', 'w') as f:
            for i in range(len(training_losses)):
                f.write('{} {} {}\n'.format(training_losses[i], validation_losses[i][0], validation_losses[i][1]))
            
    # plt.plot(losses)
    # plt.show()
