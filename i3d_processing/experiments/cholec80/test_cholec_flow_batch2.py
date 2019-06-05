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
import input_test
import math
import numpy as np
from i3d import InceptionI3d
from utils import *

# Basic model parameters as external flags.
flags = tf.app.flags
gpu_num = 1
flags.DEFINE_integer('batch_size', 1, 'Batch size.')
flags.DEFINE_integer('num_frame_per_clib', 50, 'Nummber of frames per clib')
flags.DEFINE_integer('crop_size', 224, 'Crop_size')
flags.DEFINE_integer('rgb_channels', 3, 'Channels for input')
flags.DEFINE_integer('classics', 7, 'The num of class')
FLAGS = flags.FLAGS

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def run_training():
    # Get the sets of images and labels for training, validation, and
    # Tell TensorFlow that the model will be built into the default Graph.
    pre_model_save_dir = "./models/rgb_imagenet_batch2_resume_next"
    test_list_file = '../../list/chollec80_processed_list_test_flow_full_batch2.txt'
    file = list(open(test_list_file, 'r'))
    num_test_videos = len(file)
    print("Number of test videos={}".format(num_test_videos))
    with tf.Graph().as_default():
        rgb_images_placeholder, _, labels_placeholder, is_training = placeholder_inputs(
                        FLAGS.batch_size * gpu_num,
                        FLAGS.num_frame_per_clib,
                        FLAGS.crop_size,
                        FLAGS.rgb_channels
                        )

        with tf.variable_scope('FLOW'):
            logit, _ = InceptionI3d(
                                num_classes=FLAGS.classics,
                                spatial_squeeze=True,
                                final_endpoint='Logits',
                                name='inception_i3d'
                                )(rgb_images_placeholder, is_training)
        norm_score = tf.nn.softmax(logit)

        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver()
        init = tf.global_variables_initializer()

        # Create a session for running Ops on the Graph.
        sess = tf.Session(
                        config=tf.ConfigProto(allow_soft_placement=True)
                        )
        sess.run(init)

    ckpt = tf.train.get_checkpoint_state(pre_model_save_dir)
    if ckpt and ckpt.model_checkpoint_path:
        print ("loading checkpoint %s,waiting......" % ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
        print ("load complete!")

    all_steps = num_test_videos
    top1_list = []
    for step in xrange(all_steps):
        start_time = time.time()
        s_index = 0
        predicts = []
        top1 = False
        val_images, _, val_labels, exists = input_test.import_label_flow_batch2(
                        filename=test_list_file,
                        batch_size=FLAGS.batch_size * gpu_num,
                        current_sample=step
                        )

        if (exists == 1):
            predict = sess.run(norm_score,
                               feed_dict={
                                            rgb_images_placeholder: val_images,
                                            labels_placeholder: val_labels,
                                            is_training: False
                                            })
            predicts.append(np.array(predict).astype(np.float32).reshape(FLAGS.classics))
            avg_pre = np.mean(predicts, axis=0).tolist()
            top1_test = avg_pre.index(max(avg_pre))
            top1 = (top1_test==val_labels)
            top1_list.append(top1)
            duration = time.time() - start_time

            print (predict)
            print (top1_test)
            print (top1)
            print (val_labels)
            print("Test step %d done" % (step))

            print('TOP_1_ACC in test: %f , time use: %.3f' % (top1, duration))
            print(len(top1_list))
            print('TOP_1_ACC in test: %f' % np.mean(top1_list))
            print("done")


def main(_):
    run_training()


if __name__ == '__main__':
    tf.app.run()