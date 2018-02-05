# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 20:39:38 2018

@author: Wanyu Du
"""

from keras.models import Sequential
from keras.layers import Reshape
from keras.models import Model
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape, Merge, Permute
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, Reshape, core, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution3D, MaxPooling3D, ZeroPadding3D , ZeroPadding3D , UpSampling3D
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.recurrent import LSTM
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam , SGD
from keras.layers.embeddings import Embedding
from keras.utils import np_utils
from keras.regularizers import ActivityRegularizer
from keras import backend as K
import LoadBatches


def Unet (nClasses, optimizer=None, input_width=360, input_height=480, nChannels=3): 
    
    inputs = Input((nChannels, input_height, input_width))
    
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv3)

    up1 = merge([UpSampling2D(size=(2, 2))(conv3), conv2], mode='concat', concat_axis=1)
    conv4 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up1)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv4)
    
    up2 = merge([UpSampling2D(size=(2, 2))(conv4), conv1], mode='concat', concat_axis=1)
    conv5 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv5)
    
    conv6 = Convolution2D(nClasses, 1, 1, activation='relu',border_mode='same')(conv5)
    conv6 = core.Reshape((nClasses,input_height*input_width))(conv6)
    conv6 = core.Permute((2,1))(conv6)

    conv7 = core.Activation('softmax')(conv6)

    model = Model(input=inputs, output=conv7)

    if not optimizer is None:
	    model.compile(loss="categorical_crossentropy", optimizer= optimizer, metrics=['accuracy'])
	
    return model
	

if __name__ == '__main__':
    m = Unet(nClasses=101, optimizer='rmsprop', input_height=150, input_width=150)
    print("Model output shape", m.output_shape)
    
    output_height = m.outputHeight
    output_width = m.outputWidth
    epochs=5
    train_images_path='../input/dataset1/images_prepped_train/'
    save_weights_path='../input/weights/'
    train_segs_path='../input/dataset1/annotations_prepped_train/'
    train_batch_size=10
    n_classes=101
    input_height=150
    input_width=150
    
    G = LoadBatches.imageSegmentationGenerator(train_images_path, train_segs_path, 
                                               train_batch_size, n_classes, 
                                               input_height, input_width, 
                                               output_height, output_width)
    
    for ep in range(epochs):
    	m.fit_generator(G, 30, epochs=1)
    	m.save_weights(save_weights_path + "." + str(ep))
    	m.save(save_weights_path + ".model." + str(ep))
   