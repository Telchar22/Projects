import pandas as pd
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
disease_labels = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                  'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
                  'Pleural_Thickening', 'Hernia']
path = r'P:/Project/Data_augumented/'
path_images = 'P:\\Project\\Data\\NIH\\images'

################################################################################################
#                    Need to be finished
################################################################################################
def data_frame_extractor(df, class_labels):
    file_locs = df['Path'].values.tolist()  # create list
    df = df[disease_labels]
    labels = df.to_numpy()
    return file_locs, labels

def read_and_reduce_data():
    #train_df = pd.read_csv('Data/TrainFiles/Full/Train')
    #train_df= train_df.drop(train_df[train_df['Quantity_of_findings'] == 0].sample(frac=.6).index)
    #train_df.to_csv('P:/Project/NIH_DL/Data/TrainFiles/FullAug/Train', index=None)
    train_df = pd.read_csv('P:/Project/NIH_DL/Data/TrainFiles/FullAug/Train')
    image, label = data_frame_extractor(train_df, disease_labels)
    print("Train dataframe:\n")
    for label in disease_labels:
        print (f"The class {label} has {train_df[label].sum ()} samples")



def rotate_image(r, p):
    Img = Image.open(p)
    # Rotate it by 45 degrees
    rotated = Img.rotate(r)
    rotated.show()

def parse_data():
    # Read an image
    image = tf.io.read_file('P:/Project/Data/NIH/images/00000001_000.png')
    # Decode to dense vector
    image_decoded = tf.image.decode_png(image, channels=3)
    # Resize to fixed shape
    image_resized = tf.image.resize(image_decoded, [1024, 1024])
    # Normalize it from [0, 255] to [0.0, 1.0]
    a = tf.image.per_image_standardization(image_resized)   # przetestować po powrocie
    plt.imshow (a, cmap=plt.get_cmap ('gray'))

    plt.show ()
    image_normalized = image_resized / 255.0
    plt.imshow(image_normalized, cmap=plt.get_cmap('gray'))

    plt.show()
    return image_normalized

parse_data()