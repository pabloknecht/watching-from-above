#import librairies
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Sequential, layers
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras import models

from tqdm import tqdm # to observ the progression
import numpy as np
import os
from PIL import Image

import matplotlib.pyplot as plt


def load_data(data_path:str):

    classes = {
            'AnnualCrop':0,
            'Forest':1,
            'HerbaceousVegetation':2,
            'Highway':3,
            'Industrial':4,
            'Pasture':5,
            'PermanentCrop':6,
            'Residential':7,
            'River':8,
            'SeaLake':9,
            }
    imgs = []
    labels = []
    for (cl, i) in classes.items():
        images_path = [elt for elt in os.listdir(os.path.join(data_path, cl)) if elt.find('.jpg')>0]
        for img in tqdm(images_path[:3000]):
            path = os.path.join(data_path, cl, img)
            if os.path.exists(path):
                image = Image.open(path)
                imgs.append(np.array(image))
                labels.append(i)
    X = np.array(imgs)
    num_classes = len(set(labels))
    y = to_categorical(labels, num_classes)
    return X,y


def shuffle_data(X,y, seed=None):
    '''
    Shuffle the X and y datas
    if seed is used, it will fix the random
    '''
    if seed != None :
        np.random.seed(seed)
    p = np.random.permutation(len(X))
    X, y = X[p], y[p]
    return X, y

def data_split(X,y,val_perc, test_perc):
    '''
    split of the data (test, val, train)
    '''
    first_split = int(len(X) * test_perc)
    second_split = first_split + int(len(X) * val_perc)
    X_test, X_val, X_train = X[:first_split], X[first_split:second_split], X[second_split:]
    y_test, y_val, y_train = y[:first_split], y[first_split:second_split], y[second_split:]
    return X_train, X_val, X_test, y_train, y_val, y_test


def create_basic_model():

    model = Sequential()
    model.add(Rescaling(1./255, input_shape=(64,64,3)))

    model.add(layers.Conv2D(16, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(3,3)))

    model.add(layers.Conv2D(32, kernel_size=(2,2), padding='same', activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))

    #model.add(layers.Conv2D(64, kernel_size=(2,2), padding='same', activation="relu"))
    #model.add(layers.MaxPooling2D(pool_size=(2,2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(50, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    opt = optimizers.Adam(learning_rate=1e-4)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    return model



def fit_model(model, X_train, y_train,X_val, y_val,epochs=100):
    es = EarlyStopping(monitor = 'val_accuracy',
                    patience = 5,
                    restore_best_weights = True,
                    verbose = 1,
                    )

    history = model.fit(X_train, y_train,
                                validation_data = (X_val, y_val),
                                batch_size = 32,
                                epochs=epochs,
                                callbacks=[es])
    return history



def plot_loss_accuracy(history, title=None):
    fig, ax = plt.subplots(1,2, figsize=(20,7))

    # --- LOSS ---

    ax[0].plot(history.history['loss'])
    ax[0].plot(history.history['val_loss'])
    ax[0].set_title('Model loss')
    ax[0].set_ylabel('Loss')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylim((0,3))
    ax[0].legend(['Train', 'Val'], loc='best')
    ax[0].grid(axis="x",linewidth=0.5)
    ax[0].grid(axis="y",linewidth=0.5)

    # --- ACCURACY

    ax[1].plot(history.history['accuracy'])
    ax[1].plot(history.history['val_accuracy'])
    ax[1].set_title('Model Accuracy')
    ax[1].set_ylabel('Accuracy')
    ax[1].set_xlabel('Epoch')
    ax[1].legend(['Train', 'Val'], loc='best')
    ax[1].set_ylim((0,1))
    ax[1].grid(axis="x",linewidth=0.5)
    ax[1].grid(axis="y",linewidth=0.5)

    if title:
        fig.suptitle(title)


def load_model(model_path) :
    loaded_model = models.load_model(model_path)
    return loaded_model

def predict_new_images(loaded_model, X_new):
    size = int(X_new.shape[0] ** 0.5)
    y_new = loaded_model.predict(X_new)
    y_pred_class = np.argmax(y_new, axis = 1)
    return y_pred_class.reshape((size, size))

def get_new_images(new_images_path):
    imgs = []
    images_path = os.listdir(new_images_path)
    for img in tqdm(images_path):
        path = os.path.join(new_images_path, img)
        if os.path.exists(path):
            image = Image.open(path)
            imgs.append(np.array(image))
    return np.array(imgs)


def main() :
    data_path = '../raw_data/EuroSAT'
    X, y = load_data(data_path)
    X, y = shuffle_data(X,y, 0)
    X_train, X_val, X_test, y_train, y_val, y_test = data_split(X,y,0.1,0.1)
    model_basic = create_basic_model()
    history = fit_model(model_basic, X_train, y_train,X_val, y_val,epochs=2)
    plot_loss_accuracy(history)
    evaluation = model_basic.evaluate(X_test, y_test)
    test_accuracy = evaluation[-1]
    print(f"test_accuracy = {round(test_accuracy,2)*100} %")
    model_path = '../models/my_model'
    models.save_model(model_basic, model_path)


    loaded_model = load_model(model_path)
    new_images_path =  './new_tiles'
    X_new = get_new_images(new_images_path)
    y_pred_class = predict_new_images(loaded_model, X_new)
    y_pred_class
