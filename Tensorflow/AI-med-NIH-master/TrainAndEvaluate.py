import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import random
from sklearn.metrics import roc_auc_score
import os
import warnings
import matplotlib as mpl
import numpy as np
from utils import get_roc_curve
##################################CONSTANT VARIABLES####################################################################
BATCH_SIZE = 32
IMG_SIZE = 224
BUFFER_SIZE = 1024
AUTOTUNE = tf.data.experimental.AUTOTUNE  # reduce GPU and CPU idle time
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

# tensorboard --logdir=.\Display\ver_full
disease_labels = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                  'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
                  'Pleural_Thickening', 'Hernia']
path_train = 'Data/TrainFiles/Full/Train'
img_path = r'P:/Project/Data/NIH/images/'
warnings.filterwarnings('ignore')
########################################################################################################################

METRICS = [
    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall'),
    tf.keras.metrics.AUC(curve='ROC', name='auc'),
]

mpl.rcParams['figure.figsize'] = (12, 10)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
##############################################


def print_examples(img_path):
    '''
    :param img_path: path to directory where images are stored
    :return:
    '''
    rows, columns = 3, 3
    fig = plt.figure(figsize=[30, 30])
    # random.choice(os.listdir("dir"))
    for i in range(1, 10):
        item = random.choice(os.listdir(img_path))
        img = Image.open(img_path + item)  # open image from directory
        f, e = os.path.splitext(img_path + item)
        imResize = img.resize((224, 224), Image.ANTIALIAS)  # resize image with antialias filter
        fig.add_subplot(rows, columns, i)
        plt.imshow(imResize, cmap=plt.get_cmap('gray'))
    # plt.savefig('Data/Display/example_xrays.png')
    plt.show()


def data_frame_extractor(df, class_labels):
    file_locs = df['Path'].values.tolist()  # create list
    df = df[disease_labels]
    labels = df.to_numpy()
    return file_locs, labels


def parse_data(file_loc, label):
    # Read an image
    image = tf.io.read_file(file_loc)
    # Decode to dense vector
    image_decoded = tf.image.decode_png(image, channels=3)
    # Resize to fixed shape
    image_resized = tf.image.resize(image_decoded, [IMG_SIZE, IMG_SIZE])
    # Normalize it from [0, 255] to [0.0, 1.0]
    image_normalized = image_resized / 255.0
    return image_normalized, label


def create_dataset(file_locs, labels, is_training=True):
    # Create a first dataset of file paths and labels
    dataset = tf.data.Dataset.from_tensor_slices((file_locs, labels))
    # parallelize
    dataset = dataset.map(parse_data, num_parallel_calls=AUTOTUNE)

    if is_training:
        # load it once, and keep it in memory.
        dataset = dataset.cache()
        # Shuffle the data
        dataset = dataset.shuffle(buffer_size=BUFFER_SIZE)

    # Batch the data for multiple steps
    dataset = dataset.batch(BATCH_SIZE)

    # Fetch batches in the background while the model is training.
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    return dataset

def create_testing_dataset(file_locs, labels):
    dataset = tf.data.Dataset.from_tensor_slices ((file_locs, labels))
    dataset = dataset.map (parse_data)
    dataset = dataset.batch(1)
    return dataset

def get_compiled_model(metrics=METRICS):
    '''
    Build architecture based on pretrained DenseNet
    :return: compiled model
    '''
    base_model = tf.keras.applications.DenseNet121(input_shape=IMG_SHAPE,
                                                   include_top=False,
                                                   weights='imagenet')
    base_model.trainable = False

    inputs = base_model.inputs

    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    predictions = tf.keras.layers.Dense (len(disease_labels), activation="sigmoid") (x)
    model = tf.keras.Model(inputs, predictions)
    base_learning_rate = 0.0001
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=[metrics])
    #model.summary()
    return model


def plot_metrics(history):
    '''
    Plot training metrics and save to file.
    :param history: training history
    :return:
    '''
    metrics = ['loss', 'auc', 'precision', 'recall']
    for n, metric in enumerate(metrics):
        name = metric.replace("_", " ").capitalize()
        plt.subplot(2, 2, n + 1)
        plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train')
        plt.plot(history.epoch, history.history['val_' + metric],
                 color=colors[0], linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
        elif metric == 'auc':
            plt.ylim([0, 1])
        else:
            plt.ylim([0, 1])

        plt.legend()
    plt.savefig('Training metrics')
    plt.show()

def prepare_training_data_sets():
    # print_examples(img_path)
    train_df = pd.read_csv (path_train)
    valid_df = pd.read_csv ('Data/TrainFiles/Full/Validate')

    X_train, y_train = data_frame_extractor(train_df, disease_labels)
    X_valid, y_valid = data_frame_extractor(valid_df, disease_labels)
    train_set = create_dataset(X_train, y_train)
    valid_set = create_dataset(X_valid, y_valid)
    return train_set, valid_set

def prepare_testing_data_sets():
    test_df = pd.read_csv ('Data/TrainFiles/Full/Test')
    X_test, y_test = data_frame_extractor (test_df, disease_labels)
    test_set = create_testing_dataset(X_test, y_test)
    return test_set


def init_training():
    '''
    Initialize training process and plot metrics.
    :return:
    '''
    train_set, valid_set = prepare_training_data_sets()
    print ("Done prep...\n")
    model = get_compiled_model()
    print ('Training...\n')

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_auc',
        verbose=1,
        patience=8,
        mode='max',
        restore_best_weights=True)

    history = model.fit (train_set,
                         epochs=20,
                         steps_per_epoch=288,
                         validation_data=valid_set,
                         validation_steps=30,
                         callbacks=[early_stopping])

    model.save_weights (r'P:\Project\NIH_DL\tmp\test2\attempt.ckpt')
    plot_metrics(history)
    print('Done :)\n')

def evaluate_network():
    '''
    Evaluate network etc.
    :return:
    '''
    dict_eval = {}
    # get testing data
    test_set = prepare_testing_data_sets()
    # get model
    model = get_compiled_model()
    # load saved weights
    model.load_weights(r'P:\Project\NIH_DL\tmp\test2\attempt.ckpt')
    # evaluate the model

    # print("Evaluate\n")
    # result = model.evaluate(test_set, verbose = 1, steps= 690)
    # dict_eval = dict(zip(model.metrics_names, result))

    print("Results: \n")
    print(dict_eval)
    predictions = model.predict(test_set, verbose = 1)
    get_roc_curve(disease_labels, predictions,test_set)

