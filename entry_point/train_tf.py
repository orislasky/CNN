# Standard Library
import argparse
import time
import subprocess
# Third Party
import os
import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow.keras.applications.resnet50 import ResNet50
# DN from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

import cv2
import numpy as np
import os
import sys
import tensorflow as tf
import awscli
from awscli.clidriver import create_clidriver

from sklearn.model_selection import train_test_split

EPOCHS = 20
IMG_WIDTH = 224
IMG_HEIGHT = 224
NUM_CATEGORIES = 2
TEST_SIZE = 0.2

# First Party
from smdebug.tensorflow import KerasHook


def between_steps_bottleneck():
    time.sleep(1)


class CustomCallback(tf.keras.callbacks.Callback):
    def on_train_batch_begin(self, batch, logs=None):
        if 10 <= batch < 20:
            between_steps_bottleneck()
            
def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    images = []
    labels=[]
    counter = 0
    for labelname in os.listdir(data_dir):
        path= data_dir + "/" + labelname 
        for pic in os.listdir(path):
            path2= data_dir + "/" + labelname + "/" + pic
            img= cv2.imread(path2)
            img_src = cv2.imread(path2,0)
            re_img=cv2.resize(img,(IMG_WIDTH,IMG_HEIGHT))
            images.append(re_img)
            labels.append(labelname)
            counter= counter+1
    print(counter) 
    return (images,labels)
def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
# Create a convolutional neural network
    model = tf.keras.models.Sequential([

         # Convolutional layer. Learn 32 filters using a 3x3 kernel
         tf.keras.layers.Conv2D(
             32, (3, 3), activation="relu", padding = "same", input_shape=(IMG_WIDTH, IMG_HEIGHT,3)
         ),


         # Max-pooling layer, using 3x3 pool size
         tf.keras.layers.MaxPooling2D(pool_size=(4, 4)),
         tf.keras.layers.Dropout(0.2),
        
        tf.keras.layers.Conv2D(
             64, (3, 3), activation="relu", padding = "same", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
         ),


         # Max-pooling layer, using 3x3 pool size
         tf.keras.layers.MaxPooling2D(pool_size=(3, 3)),
         tf.keras.layers.Dropout(0.2),
        

         # Flatten units
         tf.keras.layers.Flatten(),
         tf.keras.layers.Dropout(0.2),
       
         
        
        
         # Add a hidden layer with dropout
         tf.keras.layers.Dense(NUM_CATEGORIES*32, activation="relu"),
         tf.keras.layers.Dropout(0.2),
        
        
         # Add a hidden layer with dropout
         tf.keras.layers.Dense(NUM_CATEGORIES*16, activation="relu"),
         tf.keras.layers.Dropout(0.2),
        

        # Add a hidden layer with dropout
         tf.keras.layers.Dense(NUM_CATEGORIES*8, activation="relu"),
         tf.keras.layers.Dropout(0.2),
        
         # Add a hidden layer with dropout
         #tf.keras.layers.Dense(NUM_CATEGORIES*4, activation="relu"),
         #tf.keras.layers.Dropout(0.1),
         
         
        
         
         # Add an output layer with output units 
         tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
     ])

# Train neural network
    model.compile(
         optimizer="adam",
         loss="categorical_crossentropy",
         metrics=["accuracy"]
   )

    return model

def aws_cli(*cmd):
    old_env = dict(os.environ)
    try:

        # Environment
        env = os.environ.copy()
        env['LC_CTYPE'] = u'en_US.UTF'
        os.environ.update(env)

        # Run awscli in the same process
        exit_code = create_clidriver().main(*cmd)

        # Deal with problems
        if exit_code > 0:
            raise RuntimeError('AWS CLI exited with code {}'.format(exit_code))
    finally:
        os.environ.clear()
        os.environ.update(old_env)

def train(batch_size, epoch, model, enable_bottleneck, data_augmentation):
    callbacks = [CustomCallback()] if enable_bottleneck else []
    subprocess.call(['aws', 's3','cp', 's3://sagemaker-us-west-2-326556103682/data/', 'brain/', '--recursive'])


   
        
    images, labels = load_data("./brain")

    labels = tf.keras.utils.to_categorical(labels)
    X_train, X_valid, Y_train, Y_valid = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE)
     

    #Y_train = to_categorical(y_train, 10)
    #Y_valid = to_categorical(y_valid, 10)

    #X_train = X_train.astype("float32")
    #X_valid = X_valid.astype("float32")

    #mean_image = np.mean(X_train, axis=0)
    #X_train -= mean_image
    #X_valid -= mean_image
    #X_train /= 128.0
    #X_valid /= 128.0

    #if not data_augmentation:
    print("fitting") 
    model.fit(
        X_train,
        Y_train,
        batch_size=batch_size,
        epochs=epoch,
        validation_data=(X_valid, Y_valid),
        shuffle=True,
        )
    model.evaluate(X_valid, Y_valid, verbose=2)

    #else:
    #   datagen = ImageDataGenerator(
    #        zca_whitening=True,
    #        width_shift_range=0.1,
    #        height_shift_range=0.1,
    #        shear_range=0.0,
    #        zoom_range=0.0,
    #        channel_shift_range=0.0,
    #        fill_mode="nearest",
    #        cval=0.0,
    #        horizontal_flip=True,
    #        vertical_flip=True,
    #        validation_split=0.0,
    #    )

    #    datagen.fit(X_train)

        # Fit the model on the batches generated by datagen.flow().
     #   model.fit_generator(
     #       datagen.flow(X_train, Y_train, batch_size=batch_size),
     #       callbacks=callbacks,
     #       epochs=epoch,
     #       validation_data=(X_valid, Y_valid),
     #       workers=1,
     #   )


def main():
    _ = KerasHook(out_dir="")  # need this line so that import doesn't get removed by pre-commit
    parser = argparse.ArgumentParser(description="Train custom oridataset")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epoch", type=int, default=20)
    parser.add_argument("--data_augmentation", type=bool, default=False)
    parser.add_argument("--model_dir", type=str, default="./model_keras_ori")
    parser.add_argument("--enable_bottleneck", type=bool, default=False)
    args = parser.parse_args()

    mirrored_strategy = tf.distribute.MirroredStrategy()

    with mirrored_strategy.scope():
        model = get_model()
        # model = ResNet50(weights=None, input_shape=(32, 32, 3), classes=10)
        opt = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

        ## model.fit(x_train, y_train, epochs=EPOCHS)
    # start the training.
    train(args.batch_size, args.epoch, model, args.enable_bottleneck, args.data_augmentation)

if __name__ == "__main__":
    main()
