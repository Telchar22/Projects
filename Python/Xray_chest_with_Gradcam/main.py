# ChestXapp
from sklearn.datasets import load_files
from tqdm import tqdm
from keras.models import model_from_json
from keras.utils import np_utils
from keras.preprocessing import image
from display import display_grid
from grad import GradCAM
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications import imagenet_utils
from multiprocessing import Pool
import imutils
import cv2
import configparser, os
import numpy as np
import threading
from keras.utils import plot_model
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

def env_configuration():
    '''
    Reads config.ini file
    :return: paths to all required files
    '''
    # reading config with paths
    read_config = configparser.ConfigParser()
    read_config.read("config.ini")
    # add path form config file to variables and return them
    my_path_checkpoints = read_config.get("Section_A", "my_path_checkpoints")

    my_path_model = read_config.get("Section_A", "my_path_model")

    my_path_test_data = read_config.get("Section_A", "my_path_test_data")

    my_path_results_pos = read_config.get("Section_A", "my_path_results_pos")

    path = read_config.get("Section_A", "path")

    path_grad_grid = read_config.get("Section_A", "path_grad_grid")

    return my_path_checkpoints, my_path_model, my_path_test_data, my_path_results_pos, path, path_grad_grid

# printing paths
def print_paths(my_path_checkpoints, my_path_model, my_path_test_data,
                my_path_results_pos, path, path_grad_grid):
    '''
    print all paths to required files
    :param my_path_checkpoints:
    :param my_path_model:
    :param my_path_test_data:
    :param my_path_results_pos:
    :param path:
    :param path_grad_grid:
    :return: nothing
    '''
    print(my_path_checkpoints)
    print(my_path_model)
    print(my_path_test_data)
    print(my_path_results_pos)
    print(path)
    print(path_grad_grid)

def path_to_tensor(img_path):
    '''
    Loads images, resize them and convert to 4 dimensional tensor
    :param img_path: path to images
    :return: 4D tensor with images data
    '''
    # loads image as image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert image type to 3D tensor
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor and return it
    return np.expand_dims(x, axis=0)

def paths_tensor(data_path):
    '''
    Creates stack of data for prediction
    :param data_path:
    :return: stack of tensor of images
    '''
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(data_path)]
    return np.vstack(list_of_tensors) # create stack of 4D images

def load_dataset(path):
    '''
    :param path: path to data
    :return: arrays of data
    '''
    data = load_files(path)  # sklearn load for files from directory
    xray_files = np.array(data['filenames'])
    # converts a class vector (integers) to binary class matrix.
    xray_targets = np_utils.to_categorical(np.array(data['target']), len(data['target_names']))
    return xray_files, xray_targets

def predictions(my_path_model, my_path_checkpoints):
    '''
    Create model for predictions.
    :param my_path_model: path to json describtion of model
    :param my_path_checkpoints: path to weights for our model
    :return: model that describes architecture of neural network
    '''
    # load json and create model
    path = my_path_model
    path2 = my_path_checkpoints

    json_file = open(path, 'r')

    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights(path2)
    print("Loaded model from disk")
    model = loaded_model
    model.summary()  # print model architecture in console
    plot_model(model, to_file='model.png')
    print("Model saved")
    return model

def predict_with_gradient(dirs):
    '''
    Classify image then create heatmap (gradient). Save combination of image, gradient and gradient on image.
    :param dirs: list of paths to all data items
    :return: nothing
    '''
    # loop through all photos in dirs and predict
    for item in dirs:
        orig = cv2.imread(path+item)
        image = load_img(path+item)
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = imagenet_utils.preprocess_input(image)

        # use the network to make predictions
        preds = model.predict(image)
        i = np.argmax(preds[0])

        label = "Gradient"
        # initialize our gradient class activation map and build the heatmap
        cam = GradCAM(model, i)
        heatmap = cam.compute_heatmap(image)
        # resize the resulting heatmap to the original input image dimensions
        # and then overlay heatmap on top of the image
        heatmap = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))
        (heatmap, output) = cam.overlay_heatmap(heatmap, orig, alpha=0.5)
        # draw the predicted label on the output image
        cv2.rectangle(output, (0, 0), (340, 40), (0, 0, 0), -1)
        cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2)
        # display the original image and resulting heatmap and output image
        # to our screen
        output = np.vstack([orig, heatmap, output])
        output = imutils.resize(output, height=700)
        # cv2.imshow("Output", output)
        cv2.waitKey(0)
        cv2.imwrite(my_path_results_pos+item, output)

##########################################
# MULTITHREADING EXAMPLE NOT CONNECTED TO PROJECT
def counting_to_5():
    '''
    Count to 5 and print value at the end.
    :return: Thread name and count value.
    '''
    # count loop
    count = 0
    for i in range(5):
        count += 1
    name = threading.currentThread().getName()
    print("Thread: " + str(name)+", counted to: "+str(count))
###########################################

if __name__ == '__main__':

    my_path_checkpoints, my_path_model, my_path_test_data, my_path_results_pos, path, path_grad_grid = env_configuration()
    print_paths(my_path_checkpoints, my_path_model, my_path_test_data, my_path_results_pos, path, path_grad_grid)
    model = predictions(my_path_model, my_path_checkpoints)
    print("\r\n")
    print("\r\n")
    test_files, test_targets = load_dataset(my_path_test_data)
    test_tensors = paths_tensor(test_files).astype('float32') / 255

    path = path
    dirs = os.listdir(path)
    predict_with_gradient(dirs)
    print("\r\n")
    print("\r\n")
    # multiprocessing
    # offload tasks to different processes
    # create a 2-worker process
    pool = Pool(processes=2)
    args = [(path, 40, 10, "all_pre.png", 2), (path_grad_grid, 120, 30, "all_after.png", 2.5)]
    results = pool.starmap(display_grid, args)  # map 2 processes with given arguments

    #  threading
    for i in range(3): # initializing 4 threads
        t = threading.Thread(target = counting_to_5)
        t.start()
    print("\r\n")
    print("Done!")