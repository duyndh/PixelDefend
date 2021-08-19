# Copyright (c) Microsoft Corporation.  Licensed under the MIT license.
"""
Training PixelCNN++
Code based on OpenAI PixelCNN++
"""

import os
import sys
import time
import json
import argparse

import numpy as np
import tensorflow as tf

import pixel_cnn_pp.nn as nn
import pixel_cnn_pp.plotting as plotting
from pixel_cnn_pp.model import *
import data.cifar10_data as cifar10_data
import data.f_mnist_data as f_mnist_data

from tensorflow.python.client import device_lib
import pickle
import copy
import gc
import time

# ================================================================

STAGE = 2

STAGE_DOWNLOAD = 1
STAGE_INIT = 2

ROOT_DIR = "/content/drive/MyDrive/Colab Notebooks/PixelDefend"
DATASET_DIR = os.path.join(ROOT_DIR, "datasets")
CACHE_DIR = os.path.join(ROOT_DIR, "cache")

# ================================================================

def parse_parameters():
    parser = argparse.ArgumentParser()
    # data I/O
    parser.add_argument('-o', '--save_dir', type=str, default='results/weights/pxpp',
                        help='Location for parameter checkpoints and samples of PixelCNN++')
    parser.add_argument('-d', '--data_set', type=str,
                        default='cifar', help='Can be either cifar | f_mnist')
    parser.add_argument('-t', '--save_interval', type=int, default=20,
                        help='Every how many epochs to write checkpoint/samples?')
    parser.add_argument('-r', '--load_params', dest='load_params', action='store_true',
                        help='Restore training from previous model checkpoint?')
    # model
    parser.add_argument('-q', '--nr_resnet', type=int, default=5,
                        help='Number of residual blocks per stage of the model')
    parser.add_argument('-n', '--nr_filters', type=int, default=160,
                        help='Number of filters to use across the model. Higher = larger model.')
    parser.add_argument('-m', '--nr_logistic_mix', type=int, default=10,
                        help='Number of logistic components in the mixture. Higher = more flexible model')
    parser.add_argument('-z', '--resnet_nonlinearity', type=str, default='concat_elu',
                        help='Which nonlinearity to use in the ResNet layers. One of "concat_elu", "elu", "relu" ')
    # optimization
    parser.add_argument('-l', '--learning_rate', type=float,
                        default=0.001, help='Base learning rate')
    parser.add_argument('-e', '--lr_decay', type=float, default=0.999995,
                        help='Learning rate decay, applied every step of the optimization')
    parser.add_argument('-b', '--batch_size', type=int, default=12,
                        help='Batch size during training per GPU')
    parser.add_argument('-a', '--init_batch_size', type=int, default=100,
                        help='How much data to use for data-dependent initialization.')
    parser.add_argument('-p', '--dropout_p', type=float, default=0.5,
                        help='Dropout strength (i.e. 1 - keep_prob). 0 = No dropout, higher = more dropout.')
    parser.add_argument('-x', '--max_epochs', type=int,
                        default=5000, help='How many epochs to run in total?')
    parser.add_argument('-g', '--nr_gpu', type=int, default=4,
                        help='How many GPUs to distribute the training across?')
    # evaluation
    parser.add_argument('--polyak_decay', type=float, default=0.9995,
                        help='Exponential decay rate of the sum of previous model iterates during Polyak averaging')
    # reproducibility
    parser.add_argument('-s', '--seed', type=int, default=1,
                        help='Random seed to use')

    args = parser.parse_args()
    return args

def sample_from_model(sess):
    x_gen = [np.zeros((g_args.batch_size,) + obs_shape, dtype=np.float32) for i in range(g_args.nr_gpu)]
    for yi in range(obs_shape[0]):
        for xi in range(obs_shape[1]):
            print("> generating pixel {}, {}".format(yi, xi))
            new_x_gen_np = sess.run(new_x_gen, {t_xs[i]: x_gen[i] for i in range(g_args.nr_gpu)})
            for i in range(g_args.nr_gpu):
                x_gen[i][:, yi, xi, :] = new_x_gen_np[i][:, yi, xi, :]
    return np.concatenate(x_gen, axis=0)

# turn numpy inputs into feed_dict for use with tensorflow
def make_feed_dict(data, init=False):
    if type(data) is tuple:
        x, y = data
    else:
        x = data
        y = None
    # input to pixelCNN is scaled from uint8 [0,255] to float in range [-1,1]
    x = np.cast[np.float32]((x - 127.5) / 127.5)
    if init:
        feed_dict = {t_x_init: x}
        if y is not None:
            feed_dict.update({t_y_init: y})
    else:
        x = np.split(x, g_args.nr_gpu)
        feed_dict = {t_xs[i]: x[i] for i in range(g_args.nr_gpu)}
        if y is not None:
            y = np.split(y, g_args.nr_gpu)
            feed_dict.update({t_ys[i]: y[i] for i in range(g_args.nr_gpu)})
    return feed_dict

def dump_var(data, file_name):
    print("> dumping:", file_name)
    print(data)
    with open(os.path.join(CACHE_DIR, str(STAGE) + '_' + file_name), 'wb') as file:
        pickle.dump(data, file)
        
def load_var(file_name):
    print("> loading:", file_name)
    with open(os.path.join(CACHE_DIR, str(STAGE - 1) + '_' + file_name), 'rb') as file:
        data = pickle.load(file)
        print(data)
        return data

# ================================================================

def stage_load():

    global g_args
    global g_rng

    if STAGE == STAGE_DOWNLOAD:

        # get arguments
        g_args = parse_parameters()    

        # fix random seed for reproducibility
        g_rng = np.random.RandomState(g_args.seed) 

        # construct dataset dir for downloading
        g_args.data_dir = os.path.join(DATASET_DIR, g_args.data_set + "_clean")

    if STAGE >= STAGE_INIT:
        g_args = load_var("args")
        g_rng = load_var("rng")

        # fix dataset dir
        g_args.data_dir = os.path.join(g_args.data_dir, "cifar-10-python/cifar-10-batches-py" if g_args.data_set == "cifar" else "")

    return

def stage_dump():

    global g_args
    global g_rng

    if STAGE >= STAGE_DOWNLOAD:
        dump_var(g_args, "args")
        dump_var(g_rng, "rng")

    return

# ================================================================

print("STAGE:", STAGE)

# load
stage_load()

# get available devices
device_names = [device.name for device in device_lib.list_local_devices()]
print("device_names:", device_names)

print("args:", json.dumps(vars(g_args), indent=4,
                                  separators=(',', ':')))

tf.set_random_seed(g_args.seed)

# initialize data loaders for train/test splits
DataLoader = {'cifar': cifar10_data.DataLoader,
              'f_mnist': f_mnist_data.DataLoader}[g_args.data_set]    

# ================================================================

if STAGE == STAGE_DOWNLOAD:

    # create data directories
    for directory in [DATASET_DIR, CACHE_DIR]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    # patch some arguments
    g_args.class_conditional = False
    g_args.nr_gpu = len(device_names)

    t_train_data = DataLoader(g_args.data_dir, 'train', g_args.batch_size * g_args.nr_gpu,
                                rng=g_rng, shuffle=True, return_labels=g_args.class_conditional, download=True)
    del t_train_data
    gc.collect()

# ================================================================

elif STAGE == STAGE_INIT:

    print("> loading train data")
    train_data = DataLoader(g_args.data_dir, 'train', g_args.batch_size * g_args.nr_gpu,
                            rng=g_rng, shuffle=True, return_labels=g_args.class_conditional)
    print("> loading test data")
    test_data = DataLoader(g_args.data_dir, 'test', g_args.batch_size * g_args.nr_gpu, 
                            shuffle=False, return_labels=g_args.class_conditional)

    obs_shape = train_data.get_observation_size()  # e.g. a tuple (32,32,3) or (28,28,1)
    assert len(obs_shape) == 3, 'assumed right now'

    # data place holders
    t_x_init = tf.placeholder(tf.float32, shape=(g_args.init_batch_size,) + obs_shape)

    t_xs = [tf.placeholder(tf.float32, shape=(g_args.batch_size,) + obs_shape)
        for i in range(g_args.nr_gpu)]

    # if the model is class-conditional we'll set up label placeholders +
    # one-hot encodings 'h' to condition on
    if g_args.class_conditional:
        print("> initializing for class-conditional")

        t_num_labels = train_data.get_num_labels()
        t_y_init = tf.placeholder(tf.int32, shape=(g_args.init_batch_size,))
        t_h_init = tf.one_hot(t_y_init, t_num_labels)
        t_y_sample = np.split(
            np.mod(np.arange(g_args.batch_size * g_args.nr_gpu), t_num_labels), g_args.nr_gpu)
        t_h_sample = [tf.one_hot(tf.Variable(
            t_y_sample[i], trainable=False), t_num_labels) for i in range(g_args.nr_gpu)]
        t_ys = [tf.placeholder(tf.int32, shape=(g_args.batch_size,))
            for i in range(g_args.nr_gpu)]
        t_hs = [tf.one_hot(t_ys[i], t_num_labels) for i in range(g_args.nr_gpu)]

        del t_num_labels
        del t_y_init
        del t_y_sample
        del t_ys
        gc.collect()

    else:
        print("> initializing for non-class-conditional")

        t_h_init = None
        t_h_sample = [None] * g_args.nr_gpu
        t_hs = t_h_sample

    # If dataset is f_mnist, then reduce num_filters by 4
    if g_args.data_set == 'f_mnist':
        g_args.nr_filters //= 4

    # create the model
    t_model_opt = {'nr_resnet': g_args.nr_resnet, 'nr_filters': g_args.nr_filters,
                'nr_logistic_mix': g_args.nr_logistic_mix, 'resnet_nonlinearity': g_args.resnet_nonlinearity,
                'data_set': g_args.data_set}
    t_model = tf.make_template('model', model_spec)

    # run once for data dependent initialization of parameters
    print("> initializing for model parameters")
    t_gen_par = t_model(t_x_init, t_h_init, init=True, dropout_p=g_args.dropout_p, **t_model_opt)

    del t_x_init
    del t_h_init
    del t_gen_par
    gc.collect()

    print("OK")
    exit()

    # keep track of moving average
    t_all_params = tf.trainable_variables()
    t_ema = tf.train.ExponentialMovingAverage(decay=g_args.polyak_decay)
    t_maintain_averages_op = tf.group(t_ema.apply(t_all_params))

    # get loss gradients over multiple GPUs
    print("> getting loss gradients")
    t_grads = []
    t_loss_gen = []
    t_logits_gen = []
    t_loss_gen_test = []
    t_logits_gen_test = []
    for i in range(g_args.nr_gpu):
        with tf.device(device_names[i]):
            # train
            t_gen_par = t_model(t_xs[i], t_hs[i], ema=None, dropout_p=g_args.dropout_p, **t_model_opt)
            t_logits_gen.append(nn.mix_logistic_to_logits(t_xs[i], t_gen_par, data_set=g_args.data_set))
            t_loss_gen.append(nn.xent_from_softmax(t_xs[i], t_gen_par, data_set=g_args.data_set))
            del t_gen_par

            # gradients
            t_grads.append(tf.gradients(t_loss_gen[i], t_all_params))
            # test
            t_gen_par = t_model(t_xs[i], t_hs[i], ema=t_ema, dropout_p=0., **t_model_opt)
            t_logits_gen_test.append(nn.mix_logistic_to_logits(t_xs[i], t_gen_par, data_set=g_args.data_set))
            t_loss_gen_test.append(nn.xent_from_softmax(t_xs[i], t_gen_par, data_set=g_args.data_set))
            del t_gen_par

    del t_hs
    del t_logits_gen
    del t_logits_gen_test

    # add losses and gradients together and get training updates
    print("> adding losses and gradients")
    t_tf_lr = tf.placeholder(tf.float32, shape=[])
    with tf.device(device_names[0]):
        for i in range(1, g_args.nr_gpu):
            t_loss_gen[0] += t_loss_gen[i]
            t_loss_gen_test[0] += t_loss_gen_test[i]
            for j in range(len(t_grads[0])):
                t_grads[0][j] += t_grads[i][j]
        # training op
        t_optimizer = tf.group(nn.adam_updates(
            t_all_params, t_grads[0], lr=t_tf_lr, mom1=0.95, mom2=0.9995), t_maintain_averages_op)

    del t_all_params
    del t_maintain_averages_op
    del t_grads
    del t_tf_lr

    # convert loss to bits/dim
    bits_per_dim = t_loss_gen[0] / (g_args.nr_gpu * np.log(2.) * np.prod(obs_shape) * g_args.batch_size)
    del t_loss_gen

    bits_per_dim_test = t_loss_gen_test[0] / (g_args.nr_gpu * np.log(2.) * np.prod(obs_shape) * g_args.batch_size)
    del t_loss_gen_test

    # sample from the model
    print("> getting samples from model")
    new_x_gen = []
    for i in range(g_args.nr_gpu):
        with tf.device(device_names[i]):
            t_gen_par = t_model(t_xs[i], t_h_sample[i], ema=t_ema, dropout_p=0, **t_model_opt)
            sample = nn.sample_from_softmax(t_xs[i], t_gen_par, data_set=g_args.data_set)
            del t_gen_par

            new_x_gen.append(sample)

    del t_xs
    del t_h_sample
    del t_model_opt
    del t_model
    del t_ema

    # init & save
    initializer = tf.global_variables_initializer()
    saver = tf.train.Saver()

    # //////////// perform training //////////////
    if not os.path.exists(g_args.save_dir):
        os.makedirs(g_args.save_dir)
    print('starting training')
    test_bpd = []
    lr = g_args.learning_rate
    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.allow_growth = True
    with tf.Session(config=tfconfig) as sess:
        for epoch in range(g_args.max_epochs):
            begin = time.time()

            # init
            if epoch == 0:
                # manually retrieve exactly init_batch_size examples
                feed_dict = make_feed_dict(
                    train_data.next(g_args.init_batch_size), init=True)
                train_data.reset()  # rewind the iterator back to 0 to do one full epoch
                sess.run(initializer, feed_dict)
                print('initializing the model...')
                if g_args.load_params:
                    ckpt_file = g_args.save_dir + '/params_' + g_args.data_set + '.ckpt'
                    print('restoring parameters from', ckpt_file)
                    saver.restore(sess, ckpt_file)

            # train for one epoch
            train_losses = []
            for d in train_data:
                feed_dict = make_feed_dict(d)
                # forward/backward/update model on each gpu
                lr *= g_args.lr_decay
                feed_dict.update({t_tf_lr: lr})
                l, _ = sess.run([bits_per_dim, t_optimizer], feed_dict)
                train_losses.append(l)
                print("train_loss: {:.6f}".format(l))
            train_loss_gen = np.mean(train_losses)

            # compute likelihood over test data
            test_losses = []
            for d in test_data:
                feed_dict = make_feed_dict(d)
                l = sess.run(bits_per_dim_test, feed_dict)
                test_losses.append(l)
            test_loss_gen = np.mean(test_losses)
            test_bpd.append(test_loss_gen)

            # log progress to console
            print("Iteration %d, time = %ds, train bits_per_dim = %.4f, test bits_per_dim = %.4f" % (
                epoch, time.time() - begin, train_loss_gen, test_loss_gen))
            sys.stdout.flush()

            if epoch % g_args.save_interval == 0:
                # generate samples from the model
                sample_x = sample_from_model(sess)
                img_tile = plotting.img_tile(sample_x[:int(np.floor(np.sqrt(
                    g_args.batch_size * g_args.nr_gpu)) ** 2)], aspect_ratio=1.0, border_color=1.0, stretch=True)
                img = plotting.plot_img(img_tile, title=g_args.data_set + ' samples')
                plotting.plt.savefig(os.path.join(
                    g_args.save_dir, '%s_sample%d.png' % (g_args.data_set, epoch)))
                plotting.plt.close('all')

                # save params
                saver.save(sess, g_args.save_dir + '/params_' +
                        g_args.data_set + '.ckpt')
                np.savez(g_args.save_dir + '/test_bpd_' + g_args.data_set +
                        '.npz', test_bpd=np.array(test_bpd))

    del t_optimizer

# ================================================================

# dump
stage_dump()

exit()

# ================================================================

