# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Inception-v1 Inflated 3D ConvNet used for Kinetics CVPR paper.

The model is introduced in:

  Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset
  Joao Carreira, Andrew Zisserman
  https://arxiv.org/pdf/1705.07750v1.pdf.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sonnet as snt
import tensorflow as tf
from tensorflow.contrib import rnn

def I3D_LSTM(num_classes, cell_size, num_features):

  n_classes = num_classes
  n_units = cell_size
  n_features = num_features

  xplaceholder= tf.placeholder('float',[None,n_features])
  yplaceholder = tf.placeholder('float')

  layer ={ 'weights': tf.Variable(tf.random_normal([n_units, n_classes])),'bias': tf.Variable(tf.random_normal([n_classes]))}

  x = tf.split(xplaceholder, n_features, 1)

  lstm_cell = rnn.BasicLSTMCell(n_units)

  outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

  output = tf.matmul(outputs[-1], layer['weights']) + layer['bias']

  return output


