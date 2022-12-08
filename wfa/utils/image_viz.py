import matplotlib.pyplot as plt
from PIL import Image


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
