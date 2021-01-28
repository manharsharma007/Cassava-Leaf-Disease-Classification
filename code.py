import numpy as np # linear algebra
import pandas as pd # data processing
import cv2
import os
from zipfile import ZipFile
from tqdm import tqdm
import matplotlib.pyplot as plt

import tensorflow as tf
from keras import layers
from keras.preprocessing.image import img_to_array
from keras.utils import np_utils, to_categorical
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, ZeroPadding2D, BatchNormalization,Activation
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras import regularizers, optimizers
from keras.applications import DenseNet121
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight, shuffle
from PIL import Image, ImageEnhance
from keras import backend as K
import gc



BATCH_SIZE = 32
IMG_SIZE = 320

train_csv_path = "../input/cassava-leaf-disease-classification/train.csv"
images_dir_path = "../input/cassava-leaf-disease-classification/train_images"

train_csv = pd.read_csv(train_csv_path)
train_csv['label'] = train_csv['label'].astype('string')

def prepare_Images(img):
    img = cv2.imread(img, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (450, 450))
    
    #img = crop_image_from_gray(img)
    
    #img = edgeCanny(img)
    
    #img = cv2.equalizeHist(img)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.addWeighted ( img,4, cv2.GaussianBlur( img , (0,0) , 30) ,-4 ,128)
    
    return img/255.0


#Building model
densenet = DenseNet121(
    weights="../input/densenet-keras/DenseNet-BC-121-32-no-top.h5",
    include_top=False,
    input_shape=(IMG_SIZE,IMG_SIZE,3)
)

def build_model():
    model = Sequential()
    model.add(densenet)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(0.35))
    model.add(layers.Dense(256,activation='relu'))
    model.add(layers.Dropout(0.35))
    model.add(layers.Dense(256,activation='relu'))
    model.add(layers.Dropout(0.35))
    model.add(layers.Dense(64,activation='relu'))
    model.add(BatchNormalization())
    
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(5, activation='softmax'))
    
    for layer in model.layers:
        layer.trainable=True
    
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizers.Adam(lr=0.00005),
        metrics=['accuracy']
    )
    
    return model

model = build_model()

model.summary()


train_gen = ImageDataGenerator(
                                rotation_range=270,
                                width_shift_range=0.2,
                                height_shift_range=0.2,
                                brightness_range=[0.1,0.9],
                                shear_range=25,
                                zoom_range=0.3,
                                channel_shift_range=0.1,
                                horizontal_flip=True,
                                vertical_flip=True,
                                rescale=1/255,
                                validation_split=0.2
                               )
                                    
    
valid_gen = ImageDataGenerator(rescale=1/255,
                               validation_split = 0.2
                              )


train_generator = train_gen.flow_from_dataframe(
                            dataframe=train_csv,
                            directory = images_dir_path,
                            x_col = "image_id",
                            y_col = "label",
                            target_size = (IMG_SIZE, IMG_SIZE),
                            class_mode = "categorical",
                            batch_size = BATCH_SIZE,
                            shuffle = True,
                            subset = "training",

)

valid_generator = valid_gen.flow_from_dataframe(
                            dataframe=train_csv,
                            directory = images_dir_path,
                            x_col = "image_id",
                            y_col = "label",
                            target_size = (IMG_SIZE, IMG_SIZE),
                            class_mode = "categorical",
                            batch_size = BATCH_SIZE,
                            shuffle = False,
                            subset = "validation"
)


reduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.4, patience=6, 
                               verbose=1, mode='auto', epsilon=0.0001)
#model.compile(loss='categorical_crossentropy', optimizer=optimizers.adam(lr=0.0001), metrics=['accuracy'])
# patient early stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

model.fit(
    train_generator,
    validation_data = valid_generator,
    epochs=100,
    callbacks=[es, reduceLR]
)
    

preds = []
submission = pd.read_csv('../input/cassava-leaf-disease-classification/sample_submission.csv')

for image_id in submission.image_id:
    img = tf.keras.preprocessing.image.load_img('../input/cassava-leaf-disease-classification/test_images/' + image_id)
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = tf.keras.preprocessing.image.smart_resize(img, (IMG_SIZE, IMG_SIZE))
    img = tf.reshape(img, (-1, IMG_SIZE, IMG_SIZE, 3))
    prediction = model.predict(img/255)
    preds.append(np.argmax(prediction))

my_submission = pd.DataFrame({'image_id': ss.image_id, 'label': preds})
my_submission.to_csv('submission.csv', index=False) 
