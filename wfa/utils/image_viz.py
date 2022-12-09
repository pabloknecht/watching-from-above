import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import numpy as np


def plot_image_categories(img, classes):
    '''
    Plot the images with the quadrants and correspondent classes
    '''
    xs = range(64, img.height, 64)
    xt = range(20, img.height, 64)
    yt = range(44, img.height, 64)
    plt.imshow(img)
    # multiple lines all full height
    plt.vlines(x=xs, ymin=0, ymax=img.height-1, colors='red', ls='-', lw=0.5)
    plt.hlines(y=xs, xmin=0, xmax=img.height-1, colors='red', ls='-', lw=0.5)
    for item_x,value_x in enumerate(xt):
        for item_y,value_y in enumerate(yt):
            plt.text(value_x, value_y, classes[item_x, item_y], color = 'red')
    plt.show()

def plot_sub_images_categories(img, classes):
    '''
    Plot the images with the quadrants and correspondent classes
    '''

    quads = int(img.height/64)
    fig, axs = plt.subplots(quads, quads, figsize = (10, 10))
    for i in range(quads):
        for j in range(quads):
            img_quad = img.crop((i*64, j*64, i*64+64, j*64+64))
            axs[j, i].imshow(img_quad)
            axs[j, i].text(22, 40, classes[i, j], color = 'red')
            axs[j, i].axis('off')
    plt.show()

def plot_classified_images(X_new, y_pred_class):
    """
    Plot a grid of tiles contained in X_new with its correspondent predicted classes
    """
    size = int(X_new.shape[0] ** 0.5)
    X_reshaped = X_new.reshape((size,size,64,64,3))
    fig, axs = plt.subplots(size, size, figsize = (10, 10))
    for i in range(size) :
        for j in range(size) :
            axs[j, i].imshow(X_reshaped[i,j])
            axs[j, i].text(22, 40, y_pred_class[i, j], color = 'red')
            axs[j, i].axis('off')
    plt.show()

def summary(y_pred_class1:np.ndarray,
            y_pred_class2:np.ndarray):
    if y_pred_class1.shape != cat_pred2.shape:
         print('Images have different shapes, please check!')
    else:
        changes = pd.DataFrame(y_pred_class1 - y_pred_class2).applymap(lambda x: 1 if x != 0 else 0).to_numpy()

        inv_classes = {
            0:'AnnualCrop',
            1:'Forest',
            2:'HerbaceousVegetation',
            3:'Highway',
            4:'Industrial',
            5:'Pasture',
            6:'PermanentCrop',
            7:'Residential',
            8:'River',
            9:'SeaLake',
            }
        pred1 = pd.DataFrame(y_pred_class1.reshape((y_pred_class1.shape[0]*y_pred_class1.shape[1]))).value_counts(normalize=True).rename_axis('cat_ID').reset_index(name='year1')
        pred2 = pd.DataFrame(y_pred_class2.reshape((y_pred_class2.shape[0]*y_pred_class2.shape[1]))).value_counts(normalize=True).rename_axis('cat_ID').reset_index(name='year2')

        summary = pred1.merge(pred2, how='outer')
        summary.fillna(0, inplace=True)
        summary['diff'] = summary['year2']-summary['year1']
        summary['cat_name'] = summary['cat_ID'].apply(lambda x: inv_classes[x])
        summary = summary[['cat_ID', 'cat_name', 'year1', 'year2', 'diff']].set_index('cat_ID')
        summary['year1'] = pd.Series(["{0:.0f}%".format(val * 100) for val in summary['year1']], index = summary.index)
        summary['year2'] = pd.Series(["{0:.0f}%".format(val * 100) for val in summary['year2']], index = summary.index)
        summary['diff'] = pd.Series(["{0:.0f}%".format(val * 100) for val in summary['diff']], index = summary.index)
        return changes, summary
