#import librairies
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Sequential, layers
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import models
from sklearn.model_selection import train_test_split

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

def split_shuffle_data(X,y, test_perc, val_perc, seed=None) :
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_perc, random_state=42)           # split Train and Test
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_perc, random_state=42) # split Train and val
    return X_train, X_val, X_test, y_train, y_val, y_test


def create_ResNet50_model():

    model_ResNet50 = ResNet50(weights="imagenet", include_top=False, input_shape=(X_train[0].shape))

    # Set the first layers to be trainable/untrainable
    model_ResNet50.trainable = True

    model = Sequential()
    model.add(model_ResNet50)
    model.add(layers.Flatten())
    model.add(layers.Dense(49, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    opt = optimizers.Adam(learning_rate=1e-4)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    return model

def convert_to_tensor(X_train,X_val,X_test):
    #Convert array to tensor (speed conversion)
    X_train_tensor = tf.convert_to_tensor(X_train)
    X_val_tensor = tf.convert_to_tensor(X_val)
    X_test_tensor = tf.convert_to_tensor(X_test)
    return X_train_tensor, X_val_tensor, X_test_tensor

def fit_model(model_ResNet50, X_train_tensor, y_train, X_val_tensor,y_val):

  es = EarlyStopping(monitor = 'val_accuracy',
                   mode = 'max',
                   patience = 5,
                   verbose = 1,
                   restore_best_weights = True)

  history = model_ResNet50.fit(
                                X_train_tensor, y_train,
                                validation_data = (X_val_tensor,y_val),
                                epochs=100,
                                batch_size=32,
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


# def get_new_images(new_images_path):
#     imgs = []
#     images_path = sorted(os.listdir(new_images_path))
#     for img in tqdm(images_path):
#         path = os.path.join(new_images_path, img)
#         if os.path.exists(path):
#             image = Image.open(path)
#             imgs.append(np.array(image))
#     return np.array(imgs)

# def plot_classified_images(X_new, y_pred_class):
#     size = int(X_new.shape[0] ** 0.5)
#     X_reshaped = X_new.reshape((size,size,64,64,3))
#     fig, axs = plt.subplots(size, size, figsize = (10, 10))
#     for i in range(size) :
#         for j in range(size) :
#             axs[j, i].imshow(X_reshaped[i,j])
#             axs[j, i].text(22, 40, y_pred_class[i, j], color = 'red')
#             axs[j, i].axis('off')
#     plt.show()


# def main() :
#     data_path = '../raw_data/EuroSAT'
#     X, y = load_data(data_path)
#     X_train, X_val, X_test, y_train, y_val, y_test = split_shuffle_data(X,y,0.2,0.1)
#     model_ResNet50 = create_ResNet50_model()
#     history = fit_model(model_ResNet50, X_train_tensor, y_train, X_val_tensor,y_val)
#     plot_loss_accuracy(history)
#     evaluation = model_ResNet50.evaluate(X_test_tensor, y_test)
#     test_accuracy = evaluation[-1]
#     print(f"test_accuracy = {round(test_accuracy,2)*100} %")
#     model_path = '../models'
#     models.save_model(model_ResNet50, model_path)


#     loaded_model = load_model(model_path)
#     new_images_path =  './new_tiles'
#     X_new = get_new_images(new_images_path)
#     y_pred_class = predict_new_images(loaded_model, X_new)
#     plot_classified_images(X_new, y_pred_class)
