# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 16:34:34 2018

@author: Wanyu Du
"""

# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/models/fcn32s.py
# fc weights into the 1x1 convs, get_upsampling_weight 

from keras.models import *
from keras.layers import *
import LoadBatches

VGG_Weights_path = '../input/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
IMAGE_ORDERING = 'channels_last'

def FCN32(n_classes, input_height=150, input_width=150, vgg_level=3):

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
    
	vgg = Model(img_input, x)
	vgg.load_weights(VGG_Weights_path)
    
	x = Flatten(name='flatten')(x)
	x = Dense(4096, activation='relu', name='fc1')(x)
	x = Dense(4096, activation='relu', name='fc2')(x)
	x = Dense(1000 , activation='softmax', name='predictions')(x)

	o = f5
	# Deconvolution layer
	o = (Conv2D(4096, (7, 7), activation='relu', padding='same', data_format=IMAGE_ORDERING))(o)
	o = Dropout(0.5)(o)
	o = (Conv2D(4096, (1, 1), activation='relu', padding='same', data_format=IMAGE_ORDERING))(o)
	o = Dropout(0.5)(o)
    
	o = (Conv2D(n_classes, (1, 1), kernel_initializer='he_normal', data_format=IMAGE_ORDERING))(o)
	o = Conv2DTranspose(n_classes, kernel_size=(64,64), strides=(32,32), use_bias=False, data_format=IMAGE_ORDERING)(o)
	o_shape = Model(img_input, o).output_shape
	outputHeight = o_shape[1]
	outputWidth = o_shape[2]
	print("koko", o_shape)

	# Flatten the features
	o = (Reshape((-1, outputHeight*outputWidth)))(o)
	o = (Permute((2, 1)))(o)
	o = (Activation('softmax'))(o)
    
	model = Model(img_input, o)
	model.outputWidth = outputWidth
	model.outputHeight = outputHeight

	return model

if __name__ == '__main__':
    m = FCN32(n_classes=101, input_height=150, input_width=150)
    m.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
    print("Model output shape", m.output_shape)
    
    output_height = m.outputHeight
    output_width = m.outputWidth
    epochs=5
    train_images_path='../input/dataset1/images_prepped_train/'
    save_weights_path='../input/weights/'
    train_segs_path='../input/dataset1/annotations_prepped_train/'
    train_batch_size=2
    n_classes=101
    input_height=150
    input_width=150
    
    G = LoadBatches.imageSegmentationGenerator(train_images_path, train_segs_path, 
                                               train_batch_size, n_classes, 
                                               input_height, input_width, 
                                               output_height, output_width)
    
    for ep in range(epochs):
    	m.fit_generator(G, 512, epochs=1)
    	m.save_weights(save_weights_path + "." + str(ep))
    	m.save(save_weights_path + ".model." + str(ep))
   