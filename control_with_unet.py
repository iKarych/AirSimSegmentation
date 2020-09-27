import airsim

import pprint
import os
import time
import math
import tempfile
import argparse
import os
import cv2
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Conv2D, concatenate
from tensorflow.keras.layers import MaxPooling2D, UpSampling2D
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint

## class of UNet
class UNet():
    @staticmethod
    def conv_block(net, n_filters, use_batch_norm=False):
        net = Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(net)
        if use_batch_norm:
            net = BatchNormalization()(net)
        net = Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(net)
        if use_batch_norm:
            net = BatchNormalization()(net)
        return net

    @staticmethod
    def upsample_block(net, n_filters, upsample_size=(2, 2), use_batch_norm=False):
        net = UpSampling2D(upsample_size)(net)
        net = Conv2D(n_filters, 2, activation='relu', padding='same', kernel_initializer='he_normal')(net)
        if use_batch_norm:
            net = BatchNormalization()(net)
        return net

    def build_model(self, input_size=(256, 256, 1), n_layers=4, n_filters=64, use_batch_norm=False):
        inputs = Input(input_size)
        net = inputs
        down_layers = []
        for _ in range(n_layers):
            net = self.conv_block(net, n_filters, use_batch_norm)
            print(net.get_shape())
            down_layers.append(net)
            net = MaxPooling2D((2, 2), strides=2)(net)
            n_filters *= 2

        net = Dropout(0.5)(net)
        net = self.conv_block(net, n_filters, use_batch_norm)
        print(net.get_shape())

        for conv in reversed(down_layers):
            n_filters //= 2
            net = self.upsample_block(net, n_filters, (2, 2), use_batch_norm)
            print(net.get_shape())
            net = concatenate([conv, net], axis=3)
            net = self.conv_block(net, n_filters, use_batch_norm)
        
        net = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(net)
        net = Conv2D(1, 1, activation='sigmoid')(net)
        self.model = Model(inputs=inputs, outputs=net)
        self.model.summary()


    def train(self, train_set, steps_per_epoch=300, epochs=5):
        self.model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

        def scheduler(epoch):
            if epoch < 3:
                return 0.0001
            else:
                return 0.00005

        lr_schedule = LearningRateScheduler(scheduler)
        model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)

        self.model.fit(train_set, steps_per_epoch=steps_per_epoch, epochs=epochs, callbacks=[model_checkpoint, lr_schedule])

    def predict(self, img): ##calculate what is the occupanc of the road on the pic (in percent)
        prediction[prediction < 0.5] = 0
        prediction[prediction >= 0.5] = 1
        
        road = np.count_nonzero(predicted == 0)
        percent = road/predicted.size*100;
        return prediction
            
    def load_trained_model(self, weights_path):
        self.build_model(use_batch_norm=True)
        self.model.load_weights(weights_path)


## preprocess the pic so for the input of UNet
def get_test_set(image, target_size=(256, 256)):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, target_size)
    img = img / 255
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    return img


if __name__ == "__main__":
    
    ## load UNet anf weigths
    unet = UNet()
    unet.load_trained_model('unet_membrane.hdf5')
    
    ## connect to controller
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    
    ## get in the starting position
    airsim.wait_key('Press any key to takeoff')
    client.takeoffAsync().join()
    
    ## set some starting altitude
    z=5
    
    ## fly north
    for x in range(128):
        client.moveToPositionAsync(127, x, -z, 1)
        
        ## get image
        responses = client.simGetImages([airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False)])
        response = responses[0]
        
        ## preprocessing
        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8) # get numpy array
        img_rgb = img1d.reshape(response.height, response.width, 3) 
        img_mod = get_test_set(img_rgb)
        
        ## prediction
        pred_per = unet.predict(img_mod)
        
        ## if occupancy is more 10%, you're too low. If it is less, you're too high
        if pred_per > 10:
            z += 0.1
        else:
            z -= 0.1
            
    print("landing...")
    client.landAsync().join()

    print("disarming.")
    client.armDisarm(False)
    

