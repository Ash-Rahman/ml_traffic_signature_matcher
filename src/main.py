
# import tensorflow as tf
# import pandas as pd
# # import pcap
# # import subprocess

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import os
import random
import pandas as pd
import numpy as np

import csv_feature_picker
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import tensorflow as tf
mnist = tf.keras.datasets.mnist
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from keras.utils.np_utils import to_categorical, normalize
from tensorflow.python.keras import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
from tensorflow.keras.metrics import categorical_crossentropy

def main(self=None):

    #Wednesday-workingHours.pcap_ISCX.csv
    #Monday-WorkingHours.pcap_ISCX.csv
    # flow_csv = pd.read_csv("../pcap\CIC-IDS-2017\labelled_flows\Wednesday-workingHours.pcap_ISCX.csv")
    # flow_csv_2 = pd.read_csv("../pcap\CIC-IDS-2017\labelled_flows/benign\Monday-WorkingHours.pcap_ISCX.csv")
    # traffic_types = ['BENIGN', 'DoS slowloris', 'DoS Slowhttptest', 'DoS Hulk', 'DoS GoldenEye']
    traffic_types = ['BENIGN', 'DoS slowloris']
    csv_file_names = [
                    "../pcap\CIC-IDS-2017\labelled_flows\Wednesday-workingHours.pcap_ISCX.csv",
                    "../pcap\CIC-IDS-2017\labelled_flows/benign\Monday-WorkingHours.pcap_ISCX.csv"
                ]

    data = filter_data_from_csv(csv_file_names, traffic_types)

    data_x = data[0]
    labels = data[1]
    data_x = clean_data(data_x)

    # x_train, x_test, y_train, y_test = train_test_split(data_x, labels, test_size=0.33, random_state=101)
    #
    # input_function = tf.estimator.




    # drop_cols = ['Flow ID', ' Source IP', ' Source Port', ' Destination IP', ' Destination Port', ' Timestamp']
    #
    # csv_df.drop(columns=drop_cols, inplace=True)
    # df_benign = csv_df[csv_df[' Label'] == 'BENIGN']
    # df_loris = csv_df[csv_df[' Label'] == 'DoS slowloris']
    # df_benign_and_loris = df_benign.append(df_loris)
    # df_benign_and_loris = df_benign_and_loris.dropna()
    #
    # labels = df_benign_and_loris.pop(' Label')
    # encoder = LabelEncoder()
    # labels = encoder.fit_transform(labels)
    #
    # #Set really numbers beyond the range of float64 to 0
    # df_benign_and_loris[df_benign_and_loris > 1e308] = 0
    # data_x = normalize(df_benign_and_loris.to_numpy())
    # data_x = data_x.reshape(data_x.shape[0], -1)

    # lin_svc = LinearSVC()
    # lin_svc.fit(data_x, labels)

    model = Sequential([
        Dense(units=16, input_shape=(78, ), activation='relu'),
        Dense(units=32, activation='relu'),
        # Dense(units=2, activation='softmax')
        Dense(units=2, activation='sigmoid')

    ])
    model.summary()

    model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x=data_x, y=labels, batch_size=10, epochs=3, verbose=2)
    predictions = model.predict(x=data_x, batch_size=10, verbose=0)

    round_predictions = np.argmax(predictions, axis=-1)
    cm = confusion_matrix(y_true=labels, y_pred=round_predictions)
    plot_confusion_matrix(cm, traffic_types, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues)

    # benign_df = csv_df[csv_df[' Label'] == 'BENIGN']
    # slow_loris_df = csv_df[csv_df[' Label'] == 'DoS slowloris']
    # label_y = csv_df.pop(' Label')
    #
    # benign_df.drop(columns=' Label', inplace=True)
    # slow_loris_df.drop(columns=' Label', inplace=True)
    #
    # train, test = train_test_split(benign_df, test_size=0.2)
    # train, val = train_test_split(train, test_size=0.2)
    # print(len(train), 'train examples')
    # print(len(val), 'validation examples')
    # print(len(test), 'test examples')
    #
    # class_name = ['benign', 'slow_lorris']

# def df_to_dataset(dataframe, shuffle=True, batch_size=32):
#     dataframe = dataframe.copy()
#     labels = dataframe.pop(' Label')
#
#     encoder = LabelEncoder()
#     labels = encoder.fit_transform(labels)
#
#     ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
#     if shuffle:
#         ds = ds.shuffle(buffer_size=len(dataframe))
#         ds = ds.batch(batch_size)
#     return ds

def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def filter_data_from_csv(csv_file_name_list, traffic_types):
    # csv_df = pd.DataFrame(df_csv)
    dropped_columns = ['Flow ID', ' Source IP', ' Source Port', ' Destination IP', ' Destination Port', ' Timestamp']
    df_new = pd.DataFrame()

    for csv_file_name in csv_file_name_list:
        df_csv = pd.read_csv(csv_file_name)
        df_csv.drop(columns=dropped_columns, inplace=True)

        for traffic_type in traffic_types:
            try:
                df_temp = df_csv[df_csv[' Label'] == traffic_type]
                if len(df_temp) != 0:
                    df_new = df_new.append(df_temp.copy())
            except:
                pass

    df_new = df_new.dropna()
    labels = df_new.pop(' Label')
    data_x = df_new
    label_to_int = LabelEncoder()
    labels = label_to_int.fit_transform(labels)

    #Set really numbers beyond the range of float64 to 0
    # df_new[df_new > 1e308] = 0
    # print("WOWOWOW: ", df_new)
    # data_x = normalize(df_new.to_numpy())

    # data_x = data_x.reshape(data_x.shape[0], -1)

    return [data_x, labels]

def clean_data(df):
    df = df.copy()
    # remove missing values and set extemely large numbers to 0
    df = df.dropna()
    df[df > 1e308] = 0
    # convert pandas dataframe to numpy array
    df = normalize(df.to_numpy())
    df = df.reshape(df.shape[0], -1)
    return df

def sliceTestingList(list):
    slice_object = slice( 0, round(int(len(list) / 5)) )
    newList = list[slice_object]
    return newList

def sliceTrainingList(list):
    slice_object = slice( round(int(len(list)) / 5 + 1), len(list) )
    newList = list[slice_object]
    return newList


    # (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # x_train, x_test = x_train / 255.0, x_test / 255.0
    #
    # model = tf.keras.models.Sequential([
    #     tf.keras.layers.Flatten(input_shape=(28, 28)),
    #     tf.keras.layers.Dense(128, activation='relu'),
    #     tf.keras.layers.Dropout(0.2),
    #     tf.keras.layers.Dense(10, activation='softmax')
    # ])
    #
    # model.compile(optimizer='adam',
    #               loss='sparse_categorical_crossentropy',
    #               metrics=['accuracy'])
    #
    # # loss='categorical_crossentropy',
    # model.fit(x_train, y_train, epochs=5)
    # model.evaluate(x_test, y_test)

if __name__ == "__main__":
    main()
