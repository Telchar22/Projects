from numpy import random
import pandas as pd
import os
from glob import glob

def create_paths_to_images():
    '''
     Adds paths for each image in csv file
    '''
    path = 'P:\\Project\\Data\\NIH\\images'
    df = pd.read_csv('Data/WorkFiles/Data_pre')
    # create dictionary of image paths - example: '00000001_000.png' : 'P:\\Project\\Data\\NIH\\images\\00000001_000.png'
    data_image_paths = {os.path.basename(x): x for x in
                        glob(os.path.join(path, '*.png'))}
    # update dataframe with paths
    df['Path'] = df['Image Index'].map(data_image_paths.get)
    df.rename(columns={"Unnamed: 0": "Idx"},inplace=True)
    # save dataframe with paths to csv file
    df.to_csv('Data/WorkFiles/Data_feed', index=None)


def data_split():
    '''
    Split data 80%/10%/10%
    Save files that can be used to trian NN
    '''
    create_paths_to_images()
    df = pd.read_csv('Data/WorkFiles/Data_feed')
    max_value = df['Patient ID'].max() # get number of unique patients
    print(max_value)
    id_train = []
    id_val = []
    id_test = []
    # append lists by unique ID's
    for i in range(max_value):
        variable = random.randint(10)
        if variable < 8:
            id_train.append(i + 1)
        elif variable == 8:
            id_val.append(i + 1)
        elif variable == 9:
            id_test.append(i + 1)
    print(len(id_train))
    # check if ID is in list and save dataframe with selected ID's to csv file
    df.loc[df['Patient ID'].isin(id_train)].to_csv('Data/TrainFiles/Full/Train', index=None)
    df.loc[df['Patient ID'].isin(id_val)].to_csv('Data/TrainFiles/Full/Validate', index=None)
    df.loc[df['Patient ID'].isin(id_test)].to_csv('Data/TrainFiles/Full/Test', index=None)

def data_arrange():
    '''
    Reforge csv files by removing unwanted  columns. Initialize split function and add columns with quantity of
    labels for each patient.
    '''
    df = pd.read_csv('Data/WorkFiles/Data_Entry_2017.csv')

    disease_labels = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                     'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
                     'Pleural_Thickening', 'Hernia']

    df = df.loc[:, :'Patient Gender']
    df['Quantity_of_findings'] = df['Finding Labels'].apply(
        lambda x: len(x.split('|')) if (x != 'No Finding') else 0)  # split diseases names in 'Finding labels'
    # column and put their number in new column "Quantity_of.."

    # create columns with names associated with diseases names and create 0 or 1 label
    for i in disease_labels:
        df[i] = df['Finding Labels'].map(lambda x: 1 if i in x else 0)
    df.drop(['Finding Labels', 'Follow-up #', 'Patient Age', 'Patient Gender'],axis=1, inplace=True)
    df.to_csv('Data/WorkFiles/Data_pre')
    data_split()
