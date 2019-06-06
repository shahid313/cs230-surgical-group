# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Trains and Evaluates the MNIST network using a feed dictionary."""
# pylint: disable=missing-docstring
import os
import sys
sys.path.append('../../')
import time
import numpy
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import input_data
import input_test
import math
import numpy as np
from i3d import InceptionI3d
from utils import *
from tensorflow.python import pywrap_tensorflow
from numpy.random import seed
from numpy.random import randint

# Basic model parameters as external flags.
flags = tf.app.flags
resume = 1 #whether to start from scratch or resume from pre-trained
gpu_num = 1
offset = 2000 #train offset of steps already done
flags.DEFINE_float('learning_rate', 0.0001, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 100000, 'Number of steps to run trainer.')
flags.DEFINE_integer('batch_size', 1, 'Batch size.')
flags.DEFINE_integer('num_frame_per_clib', 50, 'Nummber of frames per clib')
flags.DEFINE_integer('crop_size', 224, 'Crop_size')
flags.DEFINE_integer('rgb_channels', 3, 'RGB_channels for input')
flags.DEFINE_integer('flow_channels', 2, 'FLOW_channels for input')
flags.DEFINE_integer('classics', 7, 'The num of class')
FLAGS = flags.FLAGS
train_file = '../../list/chollec80_processed_list_rgb_full_batch2.txt'
test_file = '../../list/chollec80_processed_list_test_rgb_full_batch2.txt'

os.environ["CUDA_VISIBLE_DEVICES"] = "4"


def run_training():
    # Get the sets of images and labels for training, validation, and
    # Tell TensorFlow that the model will be built into the default Graph.

    #seed RNG
    seed(1)

    if (resume == 0):
        model_save_dir = './models/rgb_imagenet_batch2_resume'
    else:
        model_save_dir = './models/rgb_imagenet_batch2_resume_next'

    # Create model directory
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    if (resume == 0):
        print("Loading pretrained original")
        rgb_pre_model_save_dir = "../../checkpoints/rgb_imagenet"
    else:
        print("Loading pretrained cholec80")
        rgb_pre_model_save_dir = "./models/rgb_imagenet_batch2_resume"

    with tf.Graph().as_default():
        global_step = tf.get_variable(
                        'global_step',
                        [],
                        initializer=tf.constant_initializer(0),
                        trainable=False
                        )
        rgb_images_placeholder, flow_images_placeholder, labels_placeholder, is_training = placeholder_inputs(
                        FLAGS.batch_size * gpu_num,
                        FLAGS.num_frame_per_clib,
                        FLAGS.crop_size,
                        FLAGS.rgb_channels,
                        FLAGS.flow_channels
                        )
        
        class_weights_placeholder = tf.placeholder(tf.float32, shape=(FLAGS.batch_size))

        learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, decay_steps=3000, decay_rate=0.1, staircase=True)
        opt_rgb = tf.train.AdamOptimizer(learning_rate)
        #opt_stable = tf.train.MomentumOptimizer(learning_rate, 0.9)
        with tf.variable_scope('RGB'):
            rgb_logit, _ = InceptionI3d(
                                    num_classes=FLAGS.classics,
                                    spatial_squeeze=True,
                                    final_endpoint='Logits'
                                    )(rgb_images_placeholder, is_training)
        rgb_loss = tower_loss_weighted(
                                rgb_logit,
                                labels_placeholder,
                                class_weights_placeholder
                                )
        accuracy = tower_acc(rgb_logit, labels_placeholder)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        #Get the set of variables we wish to update, will pass into 'compute_gradients'
        #For now just do the final inception network and final logits layer, the other layers are frozen

        new_list = []

        #train 5a
        var_branch1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'RGB/inception_i3d/Mixed_5a/Branch_1')
        new_list.append(var_branch1)

        var_branch2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'RGB/inception_i3d/Mixed_5a/Branch_2')
        new_list.append(var_branch2)

        var_branch3 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'RGB/inception_i3d/Mixed_5a/Branch_3')
        new_list.append(var_branch3)

        #train 5b
        var_branch1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'RGB/inception_i3d/Mixed_5b/Branch_1')
        new_list.append(var_branch1)

        var_branch2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'RGB/inception_i3d/Mixed_5b/Branch_2')
        new_list.append(var_branch2)

        var_branch3 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'RGB/inception_i3d/Mixed_5b/Branch_3')
        new_list.append(var_branch3)

        #train 5c
        var_branch1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'RGB/inception_i3d/Mixed_5c/Branch_1')
        new_list.append(var_branch1)

        var_branch2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'RGB/inception_i3d/Mixed_5c/Branch_2')
        new_list.append(var_branch2)

        var_branch3 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'RGB/inception_i3d/Mixed_5c/Branch_3')
        new_list.append(var_branch3)

        #train logits
        var_logits_final = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'RGB/inception_i3d/Logits')
        new_list.append(var_logits_final)

        print("Full list")
        print(new_list)

        with tf.control_dependencies(update_ops):
            rgb_grads = opt_rgb.compute_gradients(rgb_loss, var_list=new_list)
            apply_gradient_rgb = opt_rgb.apply_gradients(rgb_grads, global_step=global_step)
            train_op = tf.group(apply_gradient_rgb)
            null_op = tf.no_op()

        # Create a saver for loading trained checkpoints.
        rgb_variable_map = {}
        for variable in tf.global_variables():
            if variable.name.split('/')[0] == 'RGB' and 'Adam' not in variable.name.split('/')[-1] and variable.name.split('/')[2] != 'Logits':
                #rgb_variable_map[variable.name.replace(':0', '')[len('RGB/inception_i3d/'):]] = variable
                rgb_variable_map[variable.name.replace(':0', '')] = variable
        rgb_saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)

        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver()
        init = tf.global_variables_initializer()

        # Create a session for running Ops on the Graph.
        sess = tf.Session(
                        config=tf.ConfigProto(allow_soft_placement=True)
                        )
        sess.run(init)
        print("Initialization Done")

        # Create summary writter
        tf.summary.scalar('accuracy', accuracy)
        tf.summary.scalar('rgb_loss', rgb_loss)
        tf.summary.scalar('learning_rate', learning_rate)
        merged = tf.summary.merge_all()

    # load pre_train models
    ckpt = tf.train.get_checkpoint_state(rgb_pre_model_save_dir)
    if ckpt and ckpt.model_checkpoint_path:
        print("loading checkpoint %s,waiting......" % ckpt.model_checkpoint_path)
        rgb_saver.restore(sess, ckpt.model_checkpoint_path)
        print("load complete!")

    train_writer = tf.summary.FileWriter('./visual_logs/train_rgb_imagenet_batch2_resume', sess.graph)
    test_writer = tf.summary.FileWriter('./visual_logs/test_rgb_imagenet_batch2_resume', sess.graph)

    file = list(open(train_file, 'r'))
    num_test_videos = len(file)

    for step in xrange(FLAGS.max_steps):
        step = offset + step
        start_time = time.time()

        #Get a sample to test
        sample_a = randint(0, num_test_videos, 1)
        sample = sample_a[0]

        #if we want linear training
        sample = step

        print ("Training sample: %d" % (sample))

        #get the processed data
        rgb_train_images, flow_train_images, train_labels, exists = input_data.import_label_rgb_batch2(
                      filename=train_file,
                      batch_size=FLAGS.batch_size * gpu_num,
                      current_sample=sample
                      )

        #actually train the model
        if (exists == 1):
            #assign weights to fight class imbalance
            weight_labels = input_data.assign_class_weights(train_labels)

            sess.run(train_op, feed_dict={
                          rgb_images_placeholder: rgb_train_images,
                          labels_placeholder: train_labels,
                          class_weights_placeholder: weight_labels,
                          is_training: True
                          })

        duration = time.time() - start_time
        print('Step %d: %.3f sec' % (step, duration))

        # Save a checkpoint and evaluate the model periodically.
        #if step % 10 == 0 or (step + 1) == FLAGS.max_steps:
        if step == 0 or (step+1) % 5 == 0 or (step + 1) == FLAGS.max_steps:

            if (exists == 1):
                print('Training Data Eval:')
                summary, acc, loss_rgb = sess.run(
                                [merged, accuracy, rgb_loss],
                                feed_dict={rgb_images_placeholder: rgb_train_images,
                                           labels_placeholder: train_labels,
                                           class_weights_placeholder: weight_labels,
                                           is_training: False
                                          })
                print("accuracy: " + "{:.5f}".format(acc))
                print("rgb_loss: " + "{:.5f}".format(loss_rgb))
                train_writer.add_summary(summary, step)

            print('Validation Data Eval:')
            #TODO: Fix to select random sample from entire test list
            sample_a = randint(0, 50, 1)
            sample = sample_a[0]
            rgb_val_images, flow_val_images, val_labels, exists = input_test.import_label_rgb_batch2(
                            filename=test_file,
                            batch_size=FLAGS.batch_size * gpu_num,
                            current_sample=sample
                            )

            if (exists == 1):
                summary, acc = sess.run(
                                [merged, accuracy],
                                feed_dict={
                                            rgb_images_placeholder: rgb_val_images,
                                            labels_placeholder: val_labels,
                                            class_weights_placeholder: weight_labels,
                                            is_training: False
                                          })
                print("accuracy: " + "{:.5f}".format(acc))
                test_writer.add_summary(summary, step)
        if step == 0 or (step+1) % 5 == 0 or (step + 1) == FLAGS.max_steps:
            saver.save(sess, os.path.join(model_save_dir, 'i3d_cholec_model'), global_step=step)
    print("done")


def main(_):
    run_training()


if __name__ == '__main__':
    tf.app.run()
