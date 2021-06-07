
import matplotlib.pyplot as plt
from load_data import load_data_after_split

def patient_overlap(df_train, df_valid, df_test):
    # check if any patient appear in more like 1 data set.

    # create set of extracted patient id's for the training set
    ids_train_set = set(df_train['Image Index'].values)
    # create set of extracted patient id's for the validation set
    ids_valid_set = set(df_valid['Image Index'].values)
    # create set of extracted patient id's for the test set
    ids_test_set = set(df_test['Image Index'].values)
    # extract list of ID's that overlap in 2 sets
    patient_overlap = list(ids_train_set.intersection(ids_valid_set))
    patient_overlap2 = list(ids_train_set.intersection(ids_test_set))
    patient_overlap3 = list(ids_test_set.intersection(ids_valid_set))
    for i in (patient_overlap, patient_overlap2, patient_overlap3):
        print(f'There are {len(i)} overlaping patients.')
        if len(i) != 0:
            print('')
            print(f'These patients are in 2 datasets:')
            print(f'{i}')

def displ_splited_incl_No_Find(dict_d, name):
    # plot distribution of labels in given set (include 'No Finding'
    plt.xticks(rotation=90) # rotate names on x
    plt.bar(*zip(*dict_d.items())) # plot data from dictionary
    plt.title(f"Frequency of Each Class in {name} dataset")
    plt.savefig(f'Data/Display/freq_of_each_class_in_{name}_dataset_incld_No_Findings', bbox_inches="tight")
    plt.show()

def displ_splited_data(dict_d, name, no_find):
    # plot distribution of labels in given set
    plt.xticks(rotation=90) # rotate names on x
    plt.bar(*zip(*dict_d.items())) # plot data from dictionary
    plt.title(f"Frequency of Each Class in {name} dataset")
    plt.savefig(f'Data/Display/freq_of_each_class_in_{name}_dataset', bbox_inches="tight")
    plt.show()
    # Display data with No Finding
    dict_d.update({'No Finding': no_find})
    displ_splited_incl_No_Find(dict_d, name)

def print_data(df_train, df_valid, df_test, train_dict, valid_dict, test_dict, no_find_list):
    # printing disease frequencies in data sets
    # create list of class labels
    class_labels = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                     'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
                     'Pleural_Thickening', 'Hernia']
    # print number of samples for each class in train df
    print("Train dataframe:\n")
    for label in class_labels:
        print(f"The class {label} has {df_train[label].sum()} samples")
    # print number of samples for each class in valid df
    print("\n\nValidate dataframe:\n")
    for label in class_labels:
        print(f"The class {label} has {df_valid[label].sum()} samples")
    # print number of samples for each class in test df
    print("\n\nTest dataframe:\n")
    for label in class_labels:
        print(f"The class {label} has {df_test[label].sum()} samples")

    print(f'\nThere is large part of No Finding patients in each set:\ntrain set: {no_find_list[0]}\n validate set:'
          f' {no_find_list[1]}\n test set: {no_find_list[2]}\n')
    # call ploting function for each set of data
    for i, j, k in zip((train_dict,valid_dict,test_dict), ('train', 'validate', 'test'),[0,1,2]):
        displ_splited_data(i, j, no_find_list[k])

def initialize_all():
    train_dict, validate_dict, test_dict, no_find_list, df_train, df_valid, df_test = load_data_after_split()
    # checking patient overlap between sets
    patient_overlap(df_train, df_valid, df_test)
    print_data(df_train, df_valid, df_test, train_dict, validate_dict, test_dict, no_find_list)