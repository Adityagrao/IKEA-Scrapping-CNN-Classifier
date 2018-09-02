from keras import models, layers, optimizers
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.applications import VGG19

import time
import numpy as np

datagen = ImageDataGenerator(   
                                validation_split = 0.2, 
                                rotation_range=40,
                                width_shift_range=0.2,
                                height_shift_range=0.2,
                                shear_range=0.2,
                                zoom_range=0.2,
                                horizontal_flip=True,
                                fill_mode='nearest'
                            )

train_generator = datagen.flow_from_directory(
                                                directory='images/', 
                                                color_mode='rgb',
                                                target_size=(150, 150), 
                                                batch_size=32,
                                                subset="training")
validation_generator = datagen.flow_from_directory(
                                                directory='images/', 
                                                color_mode='rgb',
                                                target_size=(150, 150), 
                                                batch_size=32,
                                                subset="validation")


model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu',input_shape=(150, 150, 3)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(256, (3, 3), activation='relu'))
model.add(layers.Conv2D(256, (3, 3), activation='relu'))
model.add(layers.Conv2D(256, (3, 3), activation='relu'))
model.add(layers.Conv2D(256, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(2048, activation='relu'))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(64, activation='relu'))

model.add(layers.Dense(5, activation='sigmoid'))
model.summary()


Name = "Custom Model - {}".format(int(time.time()))
tensorboard = TensorBoard(log_dir="logs/{}".format(Name),write_grads=True,write_graph=True)

model.compile(loss='categorical_crossentropy',
optimizer=optimizers.Adam(lr=2e-5),
metrics=['acc'])
checkpointer = ModelCheckpoint(filepath="Custom_Model_Best_Weights.hdfs", verbose=0, save_best_only=True)
history = model.fit_generator(
                                train_generator,
                                epochs=20,
                                verbose=1,
                                steps_per_epoch=600,
                                validation_data=validation_generator,
                                validation_steps=50, 
                                callbacks =[tensorboard,checkpointer]   
                                )

#-------------------------------------------------------------------------------------------------------------

conv_base = VGG19(weights='imagenet',
include_top=False,
input_shape=(150, 150, 3))
conv_base.summary()

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(5, activation='sigmoid'))
model.summary()

model.compile(loss='categorical_crossentropy',
optimizer=optimizers.Adam(lr=2e-5),
metrics=['acc'])

Name = "VGG19 Model - {}".format(int(time.time()))
tensorboard = TensorBoard(log_dir="logs/{}".format(Name),write_grads=True, write_graph=True)
checkpointer = ModelCheckpoint(filepath="VGG16_Best_Weights.hdfs", verbose=0, save_best_only=True)
history = model.fit_generator(
train_generator,
steps_per_epoch=100,
epochs=20,
validation_data=validation_generator,
validation_steps=50,
callbacks =[tensorboard,checkpointer]
)

#-------------------------------------------------------------------------------------------------------------

