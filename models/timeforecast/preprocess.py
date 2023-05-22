import pandas as pd
import numpy as np
import glob
import os 
import pickle
from sklearn.preprocessing import MinMaxScaler

# Create a list of CSV file paths
csv_files = os.path.join("../fastStorage/2013-8/*.csv")


seq_length = 100              # n_timesteps (30s)
data_dim = 2                # n_features
output_dim = 1              # n_outputs

# Data Preprocessing
def build_dataset(data, seq_len):
    # flatten data
    dataX = []
    dataY = []
    for i in range(len(data) - seq_len):
        x = data[i:i+seq_len, :]
        y = data[i+seq_len, [-1]]
        dataX.append(x)
        dataY.append(y)

    return np.array(dataX), np.array(dataY)

def preprocess():

    dataX = np.empty((0,seq_length,data_dim), dtype=float)
    dataY = np.empty((0,output_dim), dtype=float)
    for fname in glob.glob(csv_files):
        df = pd.read_csv(fname, sep=';\t', engine='python')
        df = df[['Timestamp [ms]','Memory usage [KB]','Network received throughput [KB/s]']]

        df['Timestamp [ms]'] = pd.to_datetime(pd.to_numeric(df['Timestamp [ms]']), unit='ms')

        # Set timestamp as index
        df.set_index('Timestamp [ms]', inplace=True)

        # df = df.resample('1s').mean()

        # Remove any non-numeric values from the dataframe
        df = df.apply(pd.to_numeric, errors='coerce').dropna()

        # Remove missing values
        df.dropna(inplace=True)

        x, y = build_dataset(np.array(df), seq_length)

        if x.size>0 and y.size>0:
            dataX = np.append(dataX, x, axis=0)
            dataY = np.append(dataY, y, axis=0)

    # reshape the array to (2310, 2)
    dataX_reshaped = dataX.reshape(-1, 2)
    scalerX = MinMaxScaler()
    scalerX.fit(dataX_reshaped)
    dataX_scaled = scalerX.fit_transform(dataX_reshaped)
    dataX = dataX_scaled.reshape(dataX.shape)


    scalerY = MinMaxScaler()
    scalerY.fit(dataY)
    dataY = scalerY.fit_transform(dataY)

    # Split Train/Test set
    train_size = int(len(dataX)*0.8)
    train_data = dataX[0:train_size]
    test_data = dataX[train_size-seq_length:]

    train_label = dataY[0:train_size]
    test_label = dataY[train_size-seq_length:]

    with open('scalerX.pkl', 'wb') as f:
        pickle.dump(scalerX, f)
    
    with open('scalerY.pkl', 'wb') as f:
        pickle.dump(scalerY, f)


    return train_data, train_label, test_data, test_label, scalerY


if __name__ == "__main__":
    preprocess()



