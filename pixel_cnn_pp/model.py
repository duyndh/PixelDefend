# Copyright (c) Microsoft Corporation.  Licensed under the MIT license.
"""
The core Pixel-CNN model
"""

import tensorflow as tf
from tensorflow.contrib.framework.python.ops import arg_scope
import pixel_cnn_pp.nn as nn
import psutil
import gc

# import pickle
# import zlib
# import numpy as np

# def compress(obj):
#     return zlib.compress(pickle.dumps(obj))

# def decompress(obj):
#     return pickle.loads(zlib.decompress(obj))

# def compute_compressed_gated_resnet(compressed_element1, compressed_element2, conv):
#     if compressed_element2 is None:
#         return compress(nn.gated_resnet(decompress(compressed_element1), conv=conv))
#     else:
#         return compress(nn.gated_resnet(decompress(compressed_element1), decompress(compressed_element2), conv=conv))

# def compute_compressed_shifted_conv2d(conv, compressed_element, nr_filters, stride):
#     return compress(conv(decompress(compressed_element), num_filters=nr_filters, stride=stride))

def get_resources_info():
    return {"CPU (%)": psutil.cpu_percent(), "RAM (%)": psutil.virtual_memory().percent}

def model_spec(x, h=None, init=False, ema=None, dropout_p=0.5, nr_resnet=5, nr_filters=160, nr_logistic_mix=10,
               resnet_nonlinearity='concat_elu', data_set='cifar'):
    """
    We receive a Tensor x of shape (N,H,W,D1) (e.g. (12,32,32,3)) and produce
    a Tensor x_out of shape (N,H,W,D2) (e.g. (12,32,32,100)), where each fiber
    of the x_out tensor describes the predictive distribution for the RGB at
    that position.
    'h' is an optional N x K matrix of values to condition our generative model on
    """

    counters = {}
    with arg_scope([nn.conv2d, nn.deconv2d, nn.gated_resnet, nn.dense], counters=counters, init=init, ema=ema,
                   dropout_p=dropout_p):

        # parse resnet nonlinearity argument
        if resnet_nonlinearity == 'concat_elu':
            resnet_nonlinearity = nn.concat_elu
        elif resnet_nonlinearity == 'elu':
            resnet_nonlinearity = tf.nn.elu
        elif resnet_nonlinearity == 'relu':
            resnet_nonlinearity = tf.nn.relu
        else:
            raise ('resnet nonlinearity ' +
                   resnet_nonlinearity + ' is not supported')

        with arg_scope([nn.gated_resnet], nonlinearity=resnet_nonlinearity, h=h):

            # ////////// up pass through pixelCNN ////////
            xs = nn.int_shape(x)
            # add channel of ones to distinguish image from padding later on
            x_pad = tf.concat([x, tf.ones(xs[:-1] + [1])], 3)

            print("> initializing streams")
            print("> starting up pass")
            print(">", get_resources_info())
            
            # stream for pixels above
            t_u_list = []
            t_u_list.append(nn.down_shift(nn.down_shifted_conv2d(x_pad, num_filters=nr_filters, filter_size=[2, 3])))  # stream for pixels above

            # stream for up and to the left
            t_ul_list = []
            t_ul_list.append(nn.down_shift(nn.down_shifted_conv2d(x_pad, num_filters=nr_filters, filter_size=[1, 3])) +
                       nn.right_shift(nn.down_right_shifted_conv2d(x_pad, num_filters=nr_filters, filter_size=[2, 1])))  # stream for up and to the left

            print(">", get_resources_info())
            print("> len(t_u_list):", len(t_u_list))
            print("> len(t_ul_list):", len(t_ul_list))

            print("> getting up-pass gated_resnet (1)")
            for rep in range(nr_resnet):
                t_u_list.append(nn.gated_resnet(t_u_list[-1], conv=nn.down_shifted_conv2d))
                t_ul_list.append(nn.gated_resnet(t_ul_list[-1], t_u_list[-1], conv=nn.down_right_shifted_conv2d))

            print(">", get_resources_info())
            print("> len(t_u_list):", len(t_u_list))
            print("> len(t_ul_list):", len(t_ul_list))
    
            print("> getting up-pass shifted_conv2d (2)")
            
            t_u_list.append(nn.down_shifted_conv2d(t_u_list[-1], num_filters=nr_filters, stride=[2, 2]))
            t_ul_list.append(nn.down_right_shifted_conv2d(t_ul_list[-1], num_filters=nr_filters, stride=[2, 2]))

            print(">", get_resources_info())
            print("> len(t_u_list):", len(t_u_list))
            print("> len(t_ul_list):", len(t_ul_list))

            print("> getting up-pass gated_resnet (3)")
            
            for rep in range(nr_resnet):
                t_u_list.append(nn.gated_resnet(t_u_list[-1], conv=nn.down_shifted_conv2d))
                t_ul_list.append(nn.gated_resnet(t_ul_list[-1], t_u_list[-1], conv=nn.down_right_shifted_conv2d))

            print(">", get_resources_info())
            print("> len(t_u_list):", len(t_u_list))
            print("> len(t_ul_list):", len(t_ul_list))
            
            print("> getting up-pass shifted_conv2d (4)")
            
            t_u_list.append(nn.down_shifted_conv2d(t_u_list[-1], num_filters=nr_filters, stride=[2, 2]))
            t_ul_list.append(nn.down_right_shifted_conv2d(t_ul_list[-1], num_filters=nr_filters, stride=[2, 2]))

            print(">", get_resources_info())
            print("> len(t_u_list):", len(t_u_list))
            print("> len(t_ul_list):", len(t_ul_list))

            print("> getting up-pass gated_resnet (5)")
            
            for rep in range(nr_resnet):
                t_u_list.append(nn.gated_resnet(t_u_list[-1], conv=nn.down_shifted_conv2d))
                t_ul_list.append(nn.gated_resnet(t_ul_list[-1], t_u_list[-1], conv=nn.down_right_shifted_conv2d))

            print(">", get_resources_info())
            print("> len(t_u_list):", len(t_u_list))
            print("> len(t_ul_list):", len(t_ul_list))
            
            # /////// down pass ////////
            print("> starting down pass")

            t_u = t_u_list.pop()
            t_ul = t_ul_list.pop()

            print(">", get_resources_info())
            print("> len(t_u_list):", len(t_u_list))
            print("> len(t_ul_list):", len(t_ul_list))

            print("> getting down-pass gated_resnet (5)")
            
            for rep in range(nr_resnet):
                t_u = nn.gated_resnet(t_u, t_u_list.pop(), conv=nn.down_shifted_conv2d)
                t_ul = nn.gated_resnet(t_ul, tf.concat([t_u, t_ul_list.pop()], 3), conv=nn.down_right_shifted_conv2d)

            print(">", get_resources_info())
            print("> len(t_u_list):", len(t_u_list))
            print("> len(t_ul_list):", len(t_ul_list))
            
            print("> getting down-pass shifted_deconv2d (4)")
            
            t_u = nn.down_shifted_deconv2d(t_u, num_filters=nr_filters, stride=[2, 2])
            t_ul = nn.down_right_shifted_deconv2d(t_ul, num_filters=nr_filters, stride=[2, 2])

            print(">", get_resources_info())
            print("> len(t_u_list):", len(t_u_list))
            print("> len(t_ul_list):", len(t_ul_list))
            
            print("> getting down-pass gated_resnet (3)")
            
            for rep in range(nr_resnet + 1):
                t_u = nn.gated_resnet(t_u, t_u_list.pop(), conv=nn.down_shifted_conv2d)
                t_ul = nn.gated_resnet(t_ul, tf.concat([t_u, t_ul_list.pop()], 3), conv=nn.down_right_shifted_conv2d)

            print(">", get_resources_info())
            print("> len(t_u_list):", len(t_u_list))
            print("> len(t_ul_list):", len(t_ul_list))

            print("> getting down-pass shifted_deconv2d (2)")
            
            t_u = nn.down_shifted_deconv2d(t_u, num_filters=nr_filters, stride=[2, 2])
            t_ul = nn.down_right_shifted_deconv2d(t_ul, num_filters=nr_filters, stride=[2, 2])

            print(">", get_resources_info())
            print("> len(t_u_list):", len(t_u_list))
            print("> len(t_ul_list):", len(t_ul_list))

            print("> getting down-pass gated_resnet (1)")
            
            for rep in range(nr_resnet + 1):
                t_u = nn.gated_resnet(t_u, t_u_list.pop(), conv=nn.down_shifted_conv2d)
                t_ul = nn.gated_resnet(t_ul, tf.concat([t_u, t_ul_list.pop()], 3), conv=nn.down_right_shifted_conv2d)

            print(">", get_resources_info())
            print("> len(t_u_list):", len(t_u_list))
            print("> len(t_ul_list):", len(t_ul_list))

            del t_u
            gc.collect()

            assert len(t_u_list) == 0
            assert len(t_ul_list) == 0

            del t_u_list
            del t_ul_list
            gc.collect()

            print(">", get_resources_info())            
            
            print("> getting nin")
            if data_set == 'cifar':
                x_out = nn.nin(tf.nn.elu(t_ul), 10 * nr_logistic_mix)
            elif data_set == 'f_mnist':
                x_out = nn.nin(tf.nn.elu(t_ul), 3 * nr_logistic_mix)
            else:
                raise NotImplementedError("data_set {} not recognized".format(data_set))

            del t_ul
            gc.collect()

            print(">", get_resources_info())            

            return x_out
