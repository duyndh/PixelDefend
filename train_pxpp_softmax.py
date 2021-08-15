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
from numpy.lib.function_base import copy
import tensorflow as tf

import pixel_cnn_pp.nn as nn
import pixel_cnn_pp.plotting as plotting
from pixel_cnn_pp.model import model_spec
import data.cifar10_data as cifar10_data
import data.f_mnist_data as f_mnist_data

from tensorflow.python.client import device_lib
import pickle
import copy
import gc

STAGE_DOWNLOAD = 1
STAGE_INIT = 2
STAGE_LOSS = 3

STAGE = 2

ROOT_DIR = "/content/drive/MyDrive/Colab Notebooks/PixelDefend"
DATASET_DIR = os.path.join(ROOT_DIR, "datasets")
CACHE_DIR = os.path.join(ROOT_DIR, "cache")

for directory in [DATASET_DIR, CACHE_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)


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
    return parser.parse_args()


def dump_var(data, file_name):

    print("DUMP:", file_name)
    print(data)
    with open(os.path.join(CACHE_DIR, str(STAGE) + '_' + file_name), 'wb') as file:
        pickle.dump(data, file)
        print("dumped successfully")
        
def load_var(file_name):

    print("LOAD:", file_name)
    with open(os.path.join(CACHE_DIR, str(STAGE - 1) + '_' + file_name), 'rb') as file:
        data = pickle.load(file)
        print(data)
        print("loaded successfully")
        return data


def stage_load():

    global g_args
    global g_rng

    if STAGE >= STAGE_INIT:
        g_args = load_var("args")
        g_rng = load_var("rng")

    global g_obs_shape
    global g_h_sample
    global g_hs
    global g_model_opt

    if STAGE >= STAGE_LOSS:
        load_var(g_obs_shape)
        load_var(g_h_sample)
        load_var(g_hs)
        load_var(g_model_opt)

    return

def stage_dump():

    global g_args
    global g_rng

    if STAGE >= STAGE_DOWNLOAD:
        dump_var(g_args, "args")
        dump_var(g_rng, "rng")
        tf.set_random_seed(g_args.seed)

    global g_obs_shape
    global g_h_sample
    global g_hs
    global g_model_opt

    if STAGE >= STAGE_INIT:
        dump_var(g_obs_shape, "obs_shape")
        dump_var(g_h_sample, "h_sample")
        dump_var(g_hs, "hs")
        dump_var(g_model_opt, "model_opt")

    return

def load_train_test_data():
    DataLoader = {'cifar': cifar10_data.DataLoader,
                      'f_mnist': f_mnist_data.DataLoader}[g_args.data_set]

    train_data = DataLoader(g_args.data_dir, 'train', g_args.batch_size * g_args.nr_gpu,
                              rng=g_rng, shuffle=True, return_labels=g_args.class_conditional)
    test_data = DataLoader(g_args.data_dir, 'test', g_args.batch_size *
                             g_args.nr_gpu, shuffle=False, return_labels=g_args.class_conditional)

    return train_data, test_data


# get available devices
device_names = [device.name for device in device_lib.list_local_devices()]
print("device_names:", device_names)

print("stage:", STAGE)

# load
stage_load()

if STAGE == STAGE_DOWNLOAD:

    # get arguments
    g_args = parse_parameters()

    # patch args
    g_args.nr_gpu = len(device_names)
    g_args.class_conditional = False
    print("args:")
    print(json.dumps(vars(g_args), indent=4, separators=(',', ':')))

    # fix random seed for reproducibility
    g_rng = np.random.RandomState(g_args.seed)
    tf.set_random_seed(g_args.seed)

    # initialize data loaders for train/test splits
    DataLoader = {'cifar': cifar10_data.DataLoader,
                  'f_mnist': f_mnist_data.DataLoader}[g_args.data_set]

    g_args.data_dir = os.path.join(DATASET_DIR, g_args.data_set + "_clean")

    # download
    _ = DataLoader(g_args.data_dir, 'train', g_args.batch_size * g_args.nr_gpu,
                              rng=g_rng, shuffle=True, return_labels=g_args.class_conditional, download=True)
    del _
   
if STAGE == STAGE_INIT:

    print("args:")
    print(json.dumps(vars(g_args), indent=4, separators=(',', ':')))

    g_args.data_dir = os.path.join(
        g_args.data_dir, "cifar-10-python/cifar-10-batches-py" if g_args.data_set == "cifar" else "")

    train_data, _ = load_train_test_data()
    del _
    
    # e.g. a tuple (32,32,3) or (28,28,1)
    g_obs_shape = train_data.get_observation_size()
    assert len(g_obs_shape) == 3, 'assumed right now'

    # data place holders
    x_init = tf.placeholder(tf.float32, shape=(
        g_args.init_batch_size,) + g_obs_shape)
    # if the model is class-conditional we'll set up label placeholders +
    # one-hot encodings 'h' to condition on
    if g_args.class_conditional:
        num_labels = train_data.get_num_labels()

        y_init = tf.placeholder(tf.int32, shape=(g_args.init_batch_size,))
        h_init = tf.one_hot(y_init, num_labels)

        y_sample = np.split(
            np.mod(np.arange(g_args.batch_size * g_args.nr_gpu), num_labels), g_args.nr_gpu)
        g_h_sample = [tf.one_hot(tf.Variable(
            y_sample[i], trainable=False), num_labels) for i in range(g_args.nr_gpu)]

        ys = [tf.placeholder(tf.int32, shape=(g_args.batch_size,))
              for i in range(g_args.nr_gpu)]
        g_hs = [tf.one_hot(ys[i], num_labels) for i in range(g_args.nr_gpu)]

    else:
        h_init = None
        g_h_sample = [None] * g_args.nr_gpu
        g_hs = g_h_sample

    # If dataset is f_mnist, then reduce num_filters by 4
    if g_args.data_set == 'f_mnist':
        g_args.nr_filters //= 4

    # create the model
    g_model_opt = {'nr_resnet': g_args.nr_resnet, 'nr_filters': g_args.nr_filters,
                   'nr_logistic_mix': g_args.nr_logistic_mix, 'resnet_nonlinearity': g_args.resnet_nonlinearity,
                   'data_set': g_args.data_set}
    model = tf.make_template('model', model_spec)

    # run once for data dependent initialization of parameters
    gen_par = model(x_init, h_init, init=True,
                      dropout_p=g_args.dropout_p, **g_model_opt)

    del gen_par

# if STAGE == STAGE_LOSS:

    xs = [tf.placeholder(tf.float32, shape=(
            g_args.batch_size,) + g_obs_shape) for i in range(g_args.nr_gpu)]

    # keep track of moving average
    all_params = tf.trainable_variables()
    ema = tf.train.ExponentialMovingAverage(decay=g_args.polyak_decay)
    maintain_averages_op = tf.group(ema.apply(all_params))

    # get loss gradients over multiple GPUs
    grads = []
    loss_gen = []
    logits_gen = []
    loss_gen_test = []
    logits_gen_test = []
    for i in range(len(device_names)):
        with tf.device(device_names[i]):
            # train
            gen_par = model(xs[i], g_hs[i], ema=None,
                              dropout_p=g_args.dropout_p, **g_model_opt)
            logits_gen.append(nn.mix_logistic_to_logits(
                xs[i], gen_par, data_set=g_args.data_set))
            loss_gen.append(nn.xent_from_softmax(
                xs[i], gen_par, data_set=g_args.data_set))
            # gradients
            grads.append(tf.gradients(loss_gen[i], all_params))
            # test
            gen_par = model(xs[i], g_hs[i], ema=ema,
                              dropout_p=0., **g_model_opt)
            logits_gen_test.append(nn.mix_logistic_to_logits(
                xs[i], gen_par, data_set=g_args.data_set))
            loss_gen_test.append(nn.xent_from_softmax(
                xs[i], gen_par, data_set=g_args.data_set))

# dump
stage_dump()

# done
gc.collect()
print("DONE")
exit()

print("================================")
print("update")

# add losses and gradients together and get training updates
tf_lr = tf.placeholder(tf.float32, shape=[])
with tf.device(device_names[0]):
    for i in range(1, g_args.nr_gpu):
        loss_gen[0] += loss_gen[i]
        loss_gen_test[0] += loss_gen_test[i]
        for j in range(len(grads[0])):
            grads[0][j] += grads[i][j]
    # training op
    optimizer = tf.group(nn.adam_updates(
        all_params, grads[0], lr=tf_lr, mom1=0.95, mom2=0.9995), maintain_averages_op)

# convert loss to bits/dim
bits_per_dim = loss_gen[0] / (g_args.nr_gpu * np.log(2.)
                              * np.prod(g_obs_shape) * g_args.batch_size)
bits_per_dim_test = loss_gen_test[0] / (g_args.nr_gpu *
                                        np.log(2.) * np.prod(g_obs_shape) * g_args.batch_size)

print("================================")
print("sample")

# sample from the model
new_x_gen = []
for i in range(len(device_names)):
    with tf.device(device_names[i]):
        print(device_names[i])

        gen_par = model(xs[i], g_h_sample[i],
                          ema=ema, dropout_p=0, **g_model_opt)
        sample = nn.sample_from_softmax(
            xs[i], gen_par, data_set=g_args.data_set)
        new_x_gen.append(sample)


def sample_from_model(sess):
    x_gen = [np.zeros((g_args.batch_size,) + g_obs_shape, dtype=np.float32)
             for i in range(g_args.nr_gpu)]
    for yi in range(g_obs_shape[0]):
        for xi in range(g_obs_shape[1]):
            print("Generating pixel {}, {}".format(yi, xi))
            new_x_gen_np = sess.run(
                new_x_gen, {xs[i]: x_gen[i] for i in range(g_args.nr_gpu)})
            for i in range(g_args.nr_gpu):
                x_gen[i][:, yi, xi, :] = new_x_gen_np[i][:, yi, xi, :]
    return np.concatenate(x_gen, axis=0)


print("================================")
print("save")

# init & save
initializer = tf.global_variables_initializer()
saver = tf.train.Saver()

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
        feed_dict = {x_init: x}
        if y is not None:
            feed_dict.update({y_init: y})
    else:
        x = np.split(x, g_args.nr_gpu)
        feed_dict = {xs[i]: x[i] for i in range(g_args.nr_gpu)}
        if y is not None:
            y = np.split(y, g_args.nr_gpu)
            feed_dict.update({ys[i]: y[i] for i in range(g_args.nr_gpu)})
    return feed_dict


print("================================")
print("train")

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
            feed_dict.update({tf_lr: lr})
            l, _ = sess.run([bits_per_dim, optimizer], feed_dict)
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
            img = plotting.plot_img(
                img_tile, title=g_args.data_set + ' samples')
            plotting.plt.savefig(os.path.join(
                g_args.save_dir, '%s_sample%d.png' % (g_args.data_set, epoch)))
            plotting.plt.close('all')

            # save params
            saver.save(sess, g_args.save_dir + '/params_' +
                       g_args.data_set + '.ckpt')
            np.savez(g_args.save_dir + '/test_bpd_' + g_args.data_set +
                     '.npz', test_bpd=np.array(test_bpd))
