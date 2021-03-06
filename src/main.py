
# import tensorflow as tf
# import pandas as pd
# # import pcap
# # import subprocess
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# import tensorflow as tf
# from sklearn import preprocessing
# import os
# import random

import pandas as pd
import numpy as np


from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical, normalize
from sklearn.svm import LinearSVC
from tensorflow.python.keras import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

import model_visualiser

traffic_types = [
    'BENIGN',
    'DoS slowloris',
    'DoS Slowhttptest',
    'DoS Hulk',
    'DoS GoldenEye',
    'DDoS LOIC',
    'PortScan',
    # 'Web Attack',
    # 'Botnet ARES'
]

def main(self=None):

    #Wednesday-workingHours.pcap_ISCX.csv
    #Monday-WorkingHours.pcap_ISCX.csv
    # flow_csv = pd.read_csv("../pcap\CIC-IDS-2017\labelled_flows\Wednesday-workingHours.pcap_ISCX.csv")
    # flow_csv_2 = pd.read_csv("../pcap\CIC-IDS-2017\labelled_flows/benign\Monday-WorkingHours.pcap_ISCX.csv")

    csv_file_names = [
                         "../pcap\CIC-IDS-2017\labelled_flows\Wednesday-workingHours.pcap_ISCX.csv",
                         "../pcap\CIC-IDS-2017\labelled_flows/benign\Monday-WorkingHours.pcap_ISCX.csv",
                         "../pcap\CIC-IDS-2017\labelled_flows\Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
                         "../pcap\CIC-IDS-2017\labelled_flows\Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
                         "../pcap\CIC-IDS-2017\labelled_flows\Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
                         # "../pcap\CIC-IDS-2017\labelled_flows\Friday-WorkingHours-Morning-BotNet.pcap_ISCX.csv"
                         # "../pcap\ISCX-IDS-2012/testbed-12jun_benign.pcap_Flow.csv"
                     ]

    # Filter and clean datasets
    data = filter_data_from_csv(csv_file_names)
    data_x = data[0]
    labels = data[1]
    data_x = clean_data(data_x)

    # User options and params
    #, random_state=10
    user_batch_size = 10
    user_epochs = 5
    user_validation_set_size = 0.1
    user_learning_rate = 0.0001
    user_test_size = 0.10

    # Split the data set into training and test sets
    train_x, test_x, train_label, test_label = train_test_split(data_x, labels, test_size=user_test_size, random_state=1)

    print("flows in training set: ", str(len(train_x)))
    print("flows in testing set: ", str(len(test_x)))

    # Neural Network Layout
    model = Sequential([
        Dense(units=77, input_shape=(77, ), activation='relu'),
        Dense(units=32, activation='relu'),
        # Dense(units=2, activation='softmax')
        Dense(units=len(traffic_types), activation='sigmoid')

    ])
    model.summary()

    # Train the model with traning data
    model.compile(optimizer=Adam(learning_rate=user_learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model_history = model.fit(x=train_x, y=train_label, validation_split=user_validation_set_size, batch_size=user_batch_size, epochs=user_epochs, shuffle=True, verbose=2)

    # Evaluate the model with test data
    model.evaluate(x=test_x, y=test_label, batch_size=user_batch_size, verbose=2)
    predictions = model.predict(x=test_x, batch_size=user_batch_size, verbose=0)

    # Visualise the test results
    round_predictions = np.argmax(predictions, axis=-1)
    print(classification_report(test_label, round_predictions, target_names=traffic_types))
    model_visualiser.plot_model_epoch_history(model_history)
    cm = confusion_matrix(y_true=test_label, y_pred=round_predictions)
    model_visualiser.plot_confusion_matrix(cm, traffic_types, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues)

def filter_data_from_csv(csv_file_name_list):
    dropped_columns = ['Flow ID', 'Source IP', 'Source Port', 'Protocol', 'Destination IP', 'Destination Port', 'Timestamp']
    df_new = pd.DataFrame()

    for csv_file_name in csv_file_name_list:
        df_csv = pd.read_csv(csv_file_name, skipinitialspace=True)
        df_csv.drop(columns=dropped_columns, inplace=True)
        for traffic_type in traffic_types:
            try:
                df_temp = df_csv[df_csv['Label'] == traffic_type]
                if len(df_temp) != 0:
                    df_new = df_new.append(df_temp.copy())
            except:
                pass

    # Separate data from corresponding label
    print_num_of_traffic_types_in_df(df_new, "all datasets")
    df_new = df_new.dropna()
    df_new = df_new.sample(frac=1)
    labels = df_new.pop('Label')
    data_x = df_new
    label_to_int = LabelEncoder()
    labels = label_to_int.fit_transform(labels)

    return [data_x, labels]

def print_num_of_traffic_types_in_df(df, data_set_name):
    for traffic_type in traffic_types:
        num_of_flows = df[df['Label'] == traffic_type]
        print("number of ", traffic_type, " flows in ", data_set_name, ": ", str(len(num_of_flows)))

def clean_data(df):
    df = df.copy()
    # remove missing values and set extemely large numbers to 0
    df = df.dropna()
    df[df > 1e308] = 0
    # convert pandas dataframe to numpy array
    df = normalize(df.to_numpy())
    df = df.reshape(df.shape[0], -1)
    return df

if __name__ == "__main__":
    main()
