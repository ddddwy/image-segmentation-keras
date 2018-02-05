# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 22:09:15 2018

@author: think
"""

from keras.models import *
from keras.layers import *

VGG_Weights_path = '../input/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
IMAGE_ORDERING = 'channels_last'

def VGGSegnet(n_classes, input_height=150, input_width=150, vgg_level=3):

	img_input = Input(shape=(input_height,input_width,3))
    
	# Block 1
	x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', data_format=IMAGE_ORDERING)(img_input)
	x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', data_format=IMAGE_ORDERING)(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', data_format=IMAGE_ORDERING)(x)
	f1 = x
    
	# Block 2
	x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', data_format=IMAGE_ORDERING)(x)
	x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', data_format=IMAGE_ORDERING)(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', data_format=IMAGE_ORDERING)(x)
	f2 = x

	# Block 3
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', data_format=IMAGE_ORDERING)(x)
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', data_format=IMAGE_ORDERING)(x)
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', data_format=IMAGE_ORDERING)(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', data_format=IMAGE_ORDERING)(x)
	f3 = x

	# Block 4
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', data_format=IMAGE_ORDERING)(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', data_format=IMAGE_ORDERING)(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', data_format=IMAGE_ORDERING)(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool', data_format=IMAGE_ORDERING)(x)
	f4 = x

	# Block 5
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', data_format=IMAGE_ORDERING)(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', data_format=IMAGE_ORDERING)(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', data_format=IMAGE_ORDERING)(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool', data_format=IMAGE_ORDERING)(x)
	f5 = x
    
	vgg  = Model(img_input, x)
	vgg.load_weights(VGG_Weights_path)
    
	x = Flatten(name='flatten')(x)
	x = Dense(4096, activation='relu', name='fc1')(x)
	x = Dense(4096, activation='relu', name='fc2')(x)
	x = Dense( 1000 , activation='softmax', name='predictions')(x)

	levels = [f1 , f2 , f3 , f4 , f5 ]

	o = levels[ vgg_level ]
	
	o = (ZeroPadding2D((1,1) , data_format=IMAGE_ORDERING))(o)
	o = (Conv2D(512, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(o)
	o = (BatchNormalization())(o)

	o = (UpSampling2D((2,2), data_format=IMAGE_ORDERING))(o)
	o = (ZeroPadding2D((1,1), data_format=IMAGE_ORDERING))(o)
	o = (Conv2D(256, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(o)
	o = (BatchNormalization())(o)

	o = (UpSampling2D((2,2), data_format=IMAGE_ORDERING))(o)
	o = (ZeroPadding2D((1,1), data_format=IMAGE_ORDERING))(o)
	o = (Conv2D(128, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(o)
	o = (BatchNormalization())(o)

	o = (UpSampling2D((2,2), data_format=IMAGE_ORDERING))(o)
	o = (ZeroPadding2D((1,1), data_format=IMAGE_ORDERING))(o)
	o = (Conv2D(64, (3,3), padding='valid', data_format=IMAGE_ORDERING))(o)
	o = (BatchNormalization())(o)


	o = Conv2D(n_classes, (3, 3), padding='same', data_format=IMAGE_ORDERING)(o)
	o_shape = Model(img_input, o).output_shape
	outputHeight = o_shape[1]
	outputWidth = o_shape[2]

	o = (Reshape((-1, outputHeight*outputWidth)))(o)
	o = (Permute((2, 1)))(o)
	o = (Activation('softmax'))(o)
	model = Model(img_input, o)
	model.outputWidth = outputWidth
	model.outputHeight = outputHeight

	return model


if __name__ == '__main__':
	m = VGGSegnet(11)
	from keras.utils import plot_model
	plot_model( m , show_shapes=True , to_file='model.png')

