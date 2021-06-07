import pandas as pd

def load_dict(a, b, c, train, validate, test):
    # create dictionaries form data frames for faster data manipulation
    for i, j in zip((a, b, c), (train, validate, test)):
        i = i.iloc[:, 4:18]
        j.update(i.sum().to_dict())

    # create list of No Findings quantities for each dataframe
    no_find_list = [0, 0, 0]
    frames = [a, b, c]
    for i, j in zip(frames, [0, 1, 2]):
        no_find_list[j] = i['Quantity_of_findings'].isin([0]).sum()
    return train, validate, test, no_find_list


def load_data_after_split():
    # load data frames with data
    df_train = pd.read_csv('Data/TrainFiles/Full/Train')
    df_valid = pd.read_csv('Data/TrainFiles/Full/Validate')
    df_test = pd.read_csv('Data/TrainFiles/Full/Test')

    # initialize dictionaries
    train_dict = {}
    valid_dict = {}
    test_dict = {}

    train_dict, valid_dict, test_dict, no_find_list = load_dict(df_train, df_valid, df_test, train_dict,
                                                                valid_dict, test_dict)

    return train_dict, valid_dict, test_dict, no_find_list, df_train, df_valid, df_test
