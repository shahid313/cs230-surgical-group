'''
This program demonstrates how to train and use a deep neural network with
2D Convolutional and recurrent layers, for activity recognition in videos.
I selected the KTH dataset since it is relatively small and has activities
that are easy to learn.

Chamin Morikawa
Last updated 2017-04-15
'''

from __future__ import print_function
import numpy as np

# for file operations
import os
from PIL import Image

import keras.backend as K
from keras.models import Sequential
from keras.layers.recurrent import LSTM, GRU # you can also try using GRU layers
from keras.optimizers import RMSprop, Adadelta, adam, sgd # you can try all these optimizers
from keras.layers.convolutional import Convolution2D
#from keras.layers.core import Dense, Activation, TimeDistributedDense, Dropout, Reshape, Flatten


from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import TimeDistributed
from random import randint
import gc
# natural sorting
import re


import collections
import pandas as pd
import io
import base64
from IPython.display import HTML
import pylab
import imageio
import scipy.misc
import math
import numpy as np
from random import shuffle
import os
import sys
import matplotlib.pyplot as plt
#matplotlib inline
import seaborn as sns
import pickle
from mpl_toolkits.axes_grid1 import ImageGrid

from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn import metrics
from sklearn import preprocessing
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

import keras
from keras.models import Model, Sequential
from keras.optimizers import Adam, SGD
from keras.applications.inception_v3 import InceptionV3
from keras.applications.densenet import DenseNet121
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler, Callback, CSVLogger
from keras.preprocessing.image import ImageDataGenerator, array_to_img
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications.xception import Xception
from keras.utils import to_categorical
from keras.layers import TimeDistributed, Input, MaxPooling3D, LSTM, Dense, Conv3D, Conv2D, BatchNormalization, Flatten, Dropout, Convolution2D, Activation, MaxPooling2D, GlobalAveragePooling2D
from keras import losses


#%%


_nsre = re.compile('([0-9]+)')
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(_nsre, s)]

# use this if you want reproducibility
#np.random.seed(2016)
    

# we will be using tensorflow
K.set_image_dim_ordering('tf')

# specifiy the path to your KTH data folder
#trg_data_root = "/path/to/dataset/KTH/"
trg_data_root = "./chollec_processed_data/"

#%%


# load training or validation data
# with 25 persons in the dataset, start_index and finish_index has to be in the range [1..25]


#X_train, y_train = load_data_for_persons(trg_data_root, 1, 25, maxToAdd)

def load_data_for_persons(root_folder, video_index_arr, frames_per_clip):
    # these strings are needed for creating subfolder names
    class_labels = ["CalotTriangleDissection", "ClippingCutting", "GallbladderPackaging", "Preparation", "CleaningCoagulation", "GallbladderDissection", "GallbladderRetraction"] # 7 labels
    frame_path = ""
    frame_set_prefix = "video" # 2 digit person ID [01..25] follows
    rec_prefix = "" # seq ID [1..4] follows
    #####rec_count = 4
    rec_count = 3
    seg_prefix = "seg" # seq ID [1..4] follows
    #seg_count = 4
    seg_count = 1
    
    
    data_array = []
    classes_array = []

    # let's make a couple of loops to generate all of them
    for i in range(0, len(class_labels)):
        # class
        print("Class Data = " + str(i))
        class_folder = trg_data_root + class_labels[i]
        for video_index in video_index_arr:

            seg_folder = class_folder "/video" + video_index + "/seg1/" 

            # get the list of files
            file_list = [f for f in os.listdir(seg_folder)]
            example_size = len(file_list)

            # for larger segments, we can change the starting point to augment the data
            clip_start_index = 0
            
            
            if example_size > frames_per_clip:
                # set a random starting point but fix length - augments data, but slows training
                #clip_start_index = randint(0, (example_size - frames_per_clip))
                # sample the frames from the center
                clip_start_index = example_size/2 - frames_per_clip/2
                example_size = frames_per_clip
            
            # need natural sort before loading data
            file_list.sort(key=natural_sort_key)

            #create a list for each segment
            current_seg_temp = []
            for n in range(int(clip_start_index),int(example_size) + int(clip_start_index)):
                print("Frames in Seg # = " + str(n))
                
                file_path = seg_folder + file_list[i]
                data = np.asarray( Image.open( file_path), dtype='uint8' )
                # remove unnecessary channels
                data_gray = np.delete(data,[1,2],2)
                data_gray = data_gray.astype('float32')/255.0
                current_seg_temp.append(data_gray)

            # preprocessing
            current_seg = np.asarray(current_seg_temp)
            current_seg = current_seg.astype('float32')

            data_array.append(current_seg)
            classes_array.append(i)

    # # create one-hot vectors from output values
    classes_one_hot = np.zeros((len(classes_array), len(class_labels)))
    classes_one_hot[np.arange(len(classes_array)), classes_array] = 1
    
    # done
    #return (np.array(data_array), classes_one_hot)
    return (data_array, classes_one_hot)


# what you need to know about data, to build the model
img_rows = 120
img_cols = 160
maxToAdd = 25 # use 25 consecutive frames from each video segment, as a training sample
nb_classes = 7
class_labels = ["CalotTriangleDissection", "ClippingCutting", "GallbladderPackaging", "Preparation", "CleaningCoagulation", "GallbladderDissection", "GallbladderRetraction"] 

# build network model
print("Building model")

#%%

#define our time-distributed setup

model = Sequential()

model.add(TimeDistributed(Conv2D(4, (5, 5), strides=(2, 2), activation='relu', padding='same'), input_shape=(maxToAdd,img_rows,img_cols,1) ))
model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

model.add(TimeDistributed(Conv2D(16, (3,3), padding='same', activation='relu')))
model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

model.add(TimeDistributed(Conv2D(64, (3,3), padding='same', activation='relu')))
model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

model.add(TimeDistributed(Flatten()))


model.add(TimeDistributed(Dense(80)))
model.add(BatchNormalization()) # required for ensuring that the network learns
model.add(Activation('relu'))



model.add(LSTM(output_dim =80, activation = 'tanh'))




model.add(Dropout(0.1))
model.add(Dense(80, activation='relu'))
model.add(Dense(nb_classes, activation='softmax'))


#model.compile(loss='categorical_crossentropy', optimizer='rmsprop') #adam is faster, but you can use the others too.
model.compile(loss='categorical_crossentropy', optimizer='adam') #adam is faster, but you can use the others too.



#%%

#training parameters
batch_size = 16 # increase if your system can cope with more data
nb_epochs = 400 # I once achieved 50% accuracy with 400 epochs. Feel free to change

print ("Loading data")
# load training data
######X_train, y_train = load_data_for_persons(trg_data_root, 1, 25, maxToAdd)
X_train, y_train = load_data_for_persons(trg_data_root, ["01", "02"] maxToAdd)

# NOTE: if you can't fit all data in memory, load a few users at a time and
# use multiple epochs. I don't recommend using one user at a time, since
# it prevents good shuffling.


#%%
# perform training
print("Training")
model.fit(np.array(np.array(X_train)), y_train, batch_size=batch_size, nb_epoch=nb_epochs, shuffle=True, verbose=1)


#%%

# clean up the memory
X_train       = None
y_train       = None
X_val = None
y_val = None
gc.collect()

print("Testing")



#load_data_for_persons(root_folder, start_index, finish_index, frames_per_clip)
# load test data: in this case, person 9
X_test, y_test = load_data_for_persons(trg_data_root, ["03"] maxToAdd)

print('Total no. of testing samples used:', y_test.shape[0])

preds = model.predict(np.array(X_test))

confusion_matrix = np.zeros(shape=(y_test.shape[1],y_test.shape[1]))
accurate_count = 0.0
for i in range(0,len(preds)):
    # updating confusion matrix
    confusion_matrix[np.argmax(preds[i])][np.argmax(np.array(y_test[i]))] += 1

    # if you are not sure of the axes of the confusion matrix, try the following line
    #print ('Predicted: ', np.argmax(preds[i]), ', actual: ', np.argmax(np.array(y_val_one_hot[i])))

    # calculating overall accuracy
    if np.argmax(preds[i])==np.argmax(np.array(y_test[i])):
        accurate_count += 1

print('Validation accuracy: ', 100*accurate_count/len(preds)),' %'
print('Confusion matrix:')
print(class_labels)
print(confusion_matrix)



'''
#save the model
jsonstring  = model.to_json()
with open("KTH_LSTM.json",'wb') as f:
    f.write(jsonstring)
model.save_weights("KTH_LSTM.h5",overwrite=True)

# done.
'''