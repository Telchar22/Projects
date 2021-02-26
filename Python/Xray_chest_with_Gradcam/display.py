import matplotlib.pyplot as plt
import os
from PIL import Image
import time

def info(title):
    '''
    Print process parameters.
    :param title: name of process
    :return: ---
    '''
    # function to display process information f.e. parent and id of single process
    print(title, "\r\n")
    print('module name:', __name__, "\r\n")
    print('parent process:', os.getppid(), "\r\n")
    print('process id:', os.getpid(), "\r\n")

def display_grid(data_path, H, pH, name, s):
    '''
    Displays grid of all images in directory.
    :param data_path: path ti images
    :param H: height of figure
    :param pH: height parameter of image
    :param name: name of saved file
    :param s: delay
    :return: ---
    '''
    time.sleep(s)
    info("function display_grid")
    time.sleep(2)
    name = name
    H = H
    number_of_images = os.listdir(data_path) # quantity of images
    print("Images: " + str(number_of_images))

    rows, columns = 20, 12  # array of sub-plots
    fig = plt.figure(figsize=[20, H])  # set image size
    i = 1
    for item in number_of_images:
        img = Image.open(data_path+item) # open image from directory
        f, e = os.path.splitext(data_path + item) # only photo
        imResize = img.resize((10, pH), Image.ANTIALIAS)  # resize image with antialias filter
        fig.add_subplot(rows, columns, i)
        plt.imshow(imResize, cmap=plt.get_cmap('gray'))
        i += 1
    plt.savefig(name)
    plt.show()
