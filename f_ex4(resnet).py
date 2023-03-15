from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Conv2D, AvgPool2D, Flatten
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import ResNet50
#from keras.callbacks import EarlyStopping
from keras.callbacks import Callback

class EarlyStoppingValAcc(Callback):
    def __init__(self, monitor='val_acc', value=0.99, verbose=1):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if current > self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True
    
def load_train(path):

    train_datagen = ImageDataGenerator(validation_split=0.25, rescale=1/255., 
                                   #horizontal_flip=1,
                                   #vertical_flip=True,
                                   #rotation_range=90,
                                   #width_shift_range=0.2,
                                   #height_shift_range=0.2
                                   )

    train_datagen_flow = train_datagen.flow_from_directory(
    path, #'/datasets/fruits_small/',
    target_size=(150, 150),
    batch_size=16,
    class_mode='sparse',
    subset='training',
    seed=12345)

    return train_datagen_flow


def create_model(input_shape):
    #print('input shape is ', input_shape)
   
    backbone = ResNet50(input_shape=input_shape,
                    #weights='imagenet', 
                    weights='/datasets/keras_models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                    include_top=False)
    
    # замораживаем ResNet50 без верхушки
    #backbone.trainable = False

    model = Sequential()
    model.add(backbone)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(12, activation='softmax')) 

    optimizer = Adam(lr=0.001) 
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', 
                  metrics=['acc'])
    
    
    
    return model


def train_model(model, train_datagen_flow, val_datagen_flow, batch_size=None, 
                epochs=12,
                steps_per_epoch=None, validation_steps=None): #steps_per_epoch=80, validation_steps=40
    
    es = [EarlyStoppingValAcc(monitor='val_acc', value=0.99, verbose=1)]

    model.fit(train_datagen_flow,
              validation_data=val_datagen_flow,
              batch_size=batch_size, 
              epochs=epochs,
              steps_per_epoch=steps_per_epoch,
              validation_steps=validation_steps,
              verbose=2
              #, shuffle=True
              , callbacks=es
              )

    return model
