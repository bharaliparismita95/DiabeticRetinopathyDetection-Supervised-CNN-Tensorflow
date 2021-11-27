from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import rmsprop
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import History
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# image specifications
img_width, img_height = 150, 150  # dimension of the input image
train_data_path = '/path to the train data directory/'
validation_data_path = '/path to the validation data directory/'
n_sample_train = 24000  # number of train samples (train neural network)
n_sample_valid = 4800  # number of validation samples (target neural network)
batch_size = 100

# input format check, not necessary but as a precaution
if K.image_data_format() == 'channels_last':
    input_shape = (img_width,img_height , 3)  # 3 means RGB
else:
    input_shape = (3, img_width, img_height)  # 3 means RGB (channels_first)

# design the network
model = models.Sequential()  # sequential model

model.add(Conv2D(32, (3, 3), input_shape=input_shape))  # convolutional layer 1 with 32 filters and kernel 3*3
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))  # pooling 1 filter size 2*2

model.add(Conv2D(64, (3, 3)))  # conv 2 filter 32, kernel 3*3
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))  # pooling 2 with filter 2*2

model.add(Conv2D(128, (3, 3)))  # conv 3 filter 64 kernel 3*3
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))  # pooling 3 with filter 2*2

model.add(Conv2D(512, (3, 3)))  # conv 3 filter 64 kernel 3*3
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))  # pooling 3 with filter 2*2

model.add(Flatten())  # flatten layer

model.add(Dense(1024))  # Dense layer 1
model.add(BatchNormalization())
model.add(Activation('relu'))  # relu activation
model.add(Dropout(0.2))

model.add(Dense(1024)) # Dense layer 2
model.add(BatchNormalization())
model.add(Activation('relu'))  # relu activation
model.add(Dropout(0.2))

model.add(Dense(5))  # Dense layer 3 (output)
model.add(Activation('softmax'))  # softmax activation

# visualize the summary of the model
print(model.summary())

# model compilation
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

valid_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = valid_datagen.flow_from_directory(
    validation_data_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

# training the model
epochs = 100
model.fit(train_generator,
          steps_per_epoch=n_sample_train // batch_size,
          epochs=epochs,
          validation_data=validation_generator,
          validation_steps=n_sample_valid // batch_size,
          callbacks=[history])

# saving the model
model.save('model1.h5')

# visualizing training and validation accuracy
plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='upper left')
plt.show()

# summarize history for loss
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='upper left')
plt.show()
