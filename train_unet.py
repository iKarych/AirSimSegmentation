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

## class for generating datasets
class Dataset():
    @staticmethod
    def make_generator(dataset_folder, img_folder, aug_dict, target_size=(256, 256), batch_size=4, seed=1):
        data_gen = ImageDataGenerator(**aug_dict)
        data_gen = data_gen.flow_from_directory(
            dataset_folder,
            classes=[img_folder],
            class_mode=None,
            color_mode="grayscale",
            target_size=target_size,
            batch_size=batch_size,
            seed=seed)
        return data_gen

    ## training dataset
    def get_train_set(self, dataset, batch_size=1, target_size=(256, 256)):
        data_gen_args = dict(rotation_range=0.2,
                            width_shift_range=0.05,
                            height_shift_range=0.05,
                            shear_range=0.05,
                            zoom_range=0.05,
                            rescale=1.0 / 255,
                            horizontal_flip=True,
                            vertical_flip=True,
                            fill_mode='nearest')

        img_gen = self.make_generator(dataset, 'image', data_gen_args, target_size=target_size, batch_size=batch_size)
        mask_gen = self.make_generator(dataset, 'label', data_gen_args, target_size=target_size, batch_size=batch_size)
        data_gen = zip(img_gen, mask_gen)
        for img, mask in data_gen:
            mask[mask > 0.5] = 1
            mask[mask <= 0.5] = 0
            yield (img, mask)

    ## testing dataset
    def get_test_set(self, image_dir, target_size=(256, 256)):
        images = []
        masks = []
        for img in os.listdir(os.path.join(image_dir, 'image')):
            img = cv2.imread(os.path.join(image_dir, 'image', img), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, target_size)
            img = img / 255
            img = np.expand_dims(img, axis=-1)
            img = np.expand_dims(img, axis=0)
            images.append(img)
        for mask in os.listdir(os.path.join(image_dir, 'label')):
            mask = cv2.imread(os.path.join(image_dir, 'label', mask))
            mask = cv2.resize(mask, target_size)
            masks.append(mask)
        for img, mask in zip(images, masks):
            yield (img, mask)

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

    def predict(self, test_set):
        i = 0
        for img, label in test_set:
            prediction = self.model.predict(img)
            prediction[prediction < 0.5] = 0
            prediction[prediction >= 0.5] = 1
            img = img * 255;
            prediction = prediction * 255
            cv2.imwrite(('prediction_' + str(i).zfill(5) + '.png'), prediction[0])
            cv2.imwrite(('image_' + str(i).zfill(5) + '.png'), img[0])
            cv2.imwrite(('label_' + str(i).zfill(5) + '.png'), label)
            cv2.waitKey(0)
            i += 1
            
    def load_trained_model(self, weights_path):
        self.build_model(use_batch_norm=True)
        self.model.load_weights(weights_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='neighbor_set')
    parser.add_argument('--train', default=0)
    args = parser.parse_args()

    if args.train == 1:
        ## training unet on generated datasets
        train_set = Dataset.get_train_set(Dataset, os.path.join(args.dataset, 'train'))
        
        print('Visualize augmented examples.')
        print('Press ESC to continue.')
        for image, label in train_set:
            cv2.imshow('image', image[0])
            cv2.imshow('label', label[0])
            key = cv2.waitKey(0)
            if key == 27:
                break
        cv2.destroyAllWindows()

        unet = UNet()
        unet.build_model(use_batch_norm=True)
        unet.train(train_set)
    
    else:
        ## if already trained, load the model
        unet = UNet()
        unet.load_trained_model('unet_membrane.hdf5')
    
    ## test on testing set
    test_set = Dataset.get_test_set(Dataset, os.path.join(args.dataset, 'test'))
    unet.predict(test_set)
