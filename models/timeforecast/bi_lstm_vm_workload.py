import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

# Dataset
import os
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from preprocess import preprocess

# Visualation
import matplotlib.pyplot as plt

torch.manual_seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


seq_length = 10              # n_timesteps
data_dim = 2                # n_features
hidden_dim = 50             
output_dim = 1              # n_outputs
learning_rate = 0.01
epochs = 10
batch_size = 64

# Data Preprocessing
train_data, train_label, test_data, test_label, scalerY = preprocess()

print(len(train_data))

trainX_tensor = torch.FloatTensor(train_data).to(device)
trainY_tensor = torch.FloatTensor(train_label).to(device)

testX_tensor = torch.FloatTensor(test_data).to(device)
testY_tensor = torch.FloatTensor(test_label).to(device)

dataset = TensorDataset(trainX_tensor, trainY_tensor)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)



# LSTM Class
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, seq_len, output_dim, layers):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.output_dim = output_dim
        self.layers = layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim, bias=True)
    
    def reset_hidden_state(self):
        self.hidden = (
            torch.zeros(self.layers, self.seq_len, self.hidden_dim),
            torch.zeros(self.layers, self.seq_len, self.hidden_dim)
        )

    def forward(self, x):
        x, _status = self.lstm(x)
        x = self.fc(x[:,-1])
        return x

class BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, seq_len, output_dim, layers):
        super(BiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.output_dim = output_dim
        self.layers = layers
        self.bilstm = nn.LSTM(input_dim, hidden_dim, num_layers=layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim*2, output_dim, bias=True) # multiply hidden_dim by 2 to account for both directions of the LSTM
    
    def reset_hidden_state(self):
        self.hidden = (
            torch.zeros(self.layers*2, self.seq_len, self.hidden_dim), # multiply layers by 2 to account for both directions of the LSTM
            torch.zeros(self.layers*2, self.seq_len, self.hidden_dim)
        )

    def forward(self, x):
        x, _status = self.bilstm(x)
        x = self.fc(x[:,-1])
        return x


LSTM = LSTM(data_dim, hidden_dim, seq_length, output_dim, 1).to(device)
BiLSTM = BiLSTM(data_dim, hidden_dim, seq_length, output_dim, 1).to(device)

# Train
def train_model(model, train_df, epochs=None, lr=None, verbose=10, patience=10):
    # loss
    criterion = nn.MSELoss().to(device)
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_hist = np.zeros(epochs)
    for epoch in range(epochs):
        avg_cost = 0
        total_batch = len(train_df)

        for batch_idx, samples in enumerate(train_df):
            x_train, y_train = samples

            model.reset_hidden_state()

            # Calculate
            outputs = model(x_train)
            # cost
            loss = criterion(outputs, y_train)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_cost += loss/total_batch
        train_hist[epoch] = avg_cost

        if epoch%verbose==0:
            print('Epoch: ', '%04d' % (epoch), 'train loss: ', '{:.7f}'.format(avg_cost))
        
        if(epoch%patience==0) & (epoch!=0):
            if train_hist[epoch-patience] <= train_hist[epoch]:
                print('\n Early Stopping')
                break
    return model.eval(), train_hist

model, train_hist = train_model(BiLSTM, dataloader, epochs=epochs, lr=learning_rate, verbose=1, patience=10)

# Evaluation
with torch.no_grad():
    pred = []
    for pr in range(len(testX_tensor)):
        model.reset_hidden_state()

        predicted = model(torch.unsqueeze(testX_tensor[pr], 0))
        predicted = torch.flatten(predicted).item()
        pred.append(predicted)

    pred_inverse = scalerY.inverse_transform(np.array(pred).reshape(-1,1))
    testY_inverse = scalerY.inverse_transform(testY_tensor.cpu())

def Metric(true, pred):
    mae = np.mean(np.abs(true-pred))
    mse = np.mean(np.square(true-pred))
    rmse = np.sqrt(mse)
    return mae, mse, rmse

mae, mse, rmse = Metric(np.array(testY_tensor.cpu()).reshape(-1), np.array(pred))
print('MAE Score: ', mae)
print('MSE Score: ', mse)
print('RMSE Score: ', rmse)


# save the trained model
model_path = "./new_bi_lstm_model3.pt"
torch.save(model.state_dict(), model_path)


# target test
# length = len(test_set)
# target = np.array(test_set)[length-seq_length:]

# target = torch.FloatTensor(target).to(device)
# target = target.reshape([1, seq_length, data_dim])

# out = model(target)
# pre = torch.flatten(out).item()

# pre = round(pre, 8)
# pre_inverse = scaler_y.inverse_transform(np.array(pre).reshape(-1,1))
# print(pre_inverse.reshape([1])[0])



