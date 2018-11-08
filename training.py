# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 17:17:59 2018

@author: jonny.evans
"""

import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model 
from keras.layers import Dropout, Dense, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.utils import np_utils
from keras.optimizers import SGD
import os
import gc

plt.style.use('ggplot')

# set variables 
#define your local path here
main_folder=os.path.join(os.path.dirname(__file__))
images_folder = main_folder + '/data/celeba-dataset/img_align_celeba/'
attr_path='/data/celeba-dataset/list_attr_celeba.csv'

TRAINING_SAMPLES = 8000
VALIDATION_SAMPLES = 1600
IMG_WIDTH = 178
IMG_HEIGHT = 218
BATCH_SIZE = 16
#just one epoch to save computing time, for more accurate results increase this number
NUM_EPOCHS = 5

#what characteristic are we going to train and test for? (note if set to 'all' it'll use all of them)
ATTR='all'

# Import InceptionV3 Model
inc_model = InceptionV3(weights=main_folder+'/inceptionv3/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5',
                        include_top=False,
                        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

#inc_model.summary()

#Adding custom Layers
x = inc_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(512, activation="relu")(x)
predictions = Dense(2, activation="softmax")(x)

# create the model 
model_ = Model(inputs=inc_model.input, outputs=predictions)

# Lock initial layers to do not be trained
for layer in model_.layers[:52]:
    layer.trainable = False

# compile the model
model_.compile(optimizer=SGD(lr=0.0001, momentum=0.9)
                    , loss='categorical_crossentropy'
                    , metrics=['accuracy'])

# import the data set that include the attribute for each picture
df_attr = pd.read_csv(main_folder + attr_path)
df_attr.set_index('image_id', inplace=True)
df_attr.replace(to_replace=-1, value=0, inplace=True) #replace -1 by 0


#if i chose it to train on all attributes by setting ATTR='all', change ATTR definition here
if(ATTR=='all'):
    ATTR=list(df_attr.columns)
elif(isinstance(ATTR, str)):
    ATTR=[ATTR]

# List of available attributes
for i, j in enumerate(df_attr.columns):
    print(i, j)
    
# Recomended partition
df_partition = pd.read_csv(main_folder + '/data/celeba-dataset/list_eval_partition.csv')
df_partition.head()

# display counter by partition
# 0 -> TRAINING
# 1 -> VALIDATION
# 2 -> TEST
df_partition['partition'].value_counts().sort_index()

df_partition.set_index('image_id', inplace=True)

def load_reshape_img(fname):
    img = load_img(fname)
    x = img_to_array(img)/255.
    x = x.reshape((1,) + x.shape)

    return x

# I know, I'm not listing all the variables in the functions properly, but this a project just for fun so ¯\_(ツ)_/¯
def generate_df(df,partition, attr, num_samples):
    '''
    partition
        0 -> train
        1 -> validation
        2 -> test
    
    '''
    
    #The sample size is at most the number stated above, but at least the size of the smallest class of the dataframe. This results in some uncommon attributes (e.g. sideburns) having to train on very few samples.
    min_class_size=min(len(df[(df['partition'] == partition) & (df[attr] == 0)]),len(df[(df['partition'] == partition) & (df[attr] == 1)]) )
    sample_size=int(num_samples/2)
    if(min_class_size<int(num_samples/2)):
        sample_size=min_class_size
    
    df_ = df[(df['partition'] == partition) & (df[attr] == 0)].sample(sample_size)
    df_ = pd.concat([df_,df[(df['partition'] == partition) & (df[attr] == 1)].sample(sample_size)])

    # for Train and Validation
    if partition != 2:
        x_ = np.array([load_reshape_img(images_folder + fname) for fname in df_.index])
        x_ = x_.reshape(x_.shape[0], 218, 178, 3)
        y_ = np_utils.to_categorical(df_[attr],2)
    # for Test
    else:
        x_ = []
        y_ = []

        for index, target in df_.iterrows():
            im = cv2.imread(images_folder + index)
            im = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (IMG_WIDTH, IMG_HEIGHT)).astype(np.float32) / 255.0
            im = np.expand_dims(im, axis =0)
            x_.append(im)
            y_.append(target[attr])

    return x_, y_

def training(ATTR, train_generator,x_valid,y_valid):
    #https://keras.io/models/sequential/ fit generator
    checkpointer = ModelCheckpoint(filepath=main_folder+'/inceptionv3/attributes/weights.best.inc.'+ATTR+'.hdf5', 
                                   verbose=1, save_best_only=True,save_weights_only=True)

    hist = model_.fit_generator(train_generator
                         , validation_data = (x_valid, y_valid)
                          , steps_per_epoch= TRAINING_SAMPLES/BATCH_SIZE
                          , epochs= NUM_EPOCHS
                          , callbacks=[checkpointer]
                          , verbose=1
                        )
    return hist

def generator(ATTR,df_partition):    
    # join the partition with the chosen attribute in the same data frame
    df_par_attr = df_partition.join(df_attr[ATTR], how='inner')

    # Create Train dataframes
    x_train, y_train = generate_df(df_par_attr,0, ATTR, TRAINING_SAMPLES)

    # Create Validation dataframes
    x_valid, y_valid = generate_df(df_par_attr,1, ATTR, VALIDATION_SAMPLES)

    # define data generator with augmentations
    train_datagen =  ImageDataGenerator(
      preprocessing_function=preprocess_input,
      rotation_range=30,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
    )
    #fit it to our training data
    train_datagen.fit(x_train)
    train_generator = train_datagen.flow(x_train, y_train,batch_size=BATCH_SIZE,)
    
    del x_train, y_train
    
    return train_generator, x_valid, y_valid
#%%
#for each attribute, run the necessary functions in order to train, and then save an Inception model for the task
for attr in ATTR[15:]:
    print('Learning to recognise: ',attr,', which is attribute',ATTR.index(attr)+1,' of ',len(ATTR))
    train_generator, x_valid, y_valid=generator(attr,df_partition)
    #gotta save memory
    gc.collect()
    training(attr, train_generator,x_valid,y_valid)
    print(' ')
    print(' ')
    #gotta save memory
    gc.collect()
    
    