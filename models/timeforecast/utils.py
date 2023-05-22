import torch
import pickle
import numpy as np
import torch.nn as nn

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

with open('scalerX.pkl', 'rb') as f:
    scalerX = pickle.load(f)

with open('scalerY.pkl', 'rb') as f:
    scalerY = pickle.load(f)


seq_length = 10              # n_timesteps
data_dim = 2                # n_features
hidden_dim = 50             
output_dim = 1              # n_outputs

network_max = 1000       # Guess
memory_max = 67108864
vm_min = 1

model = BiLSTM(data_dim, hidden_dim, seq_length, output_dim, 1)
model.load_state_dict(torch.load('new_bi_lstm_model3.pt'))
model.eval()

def planning(input, now_vm):
    result = 0
    next_vm = input//network_max
    if next_vm > now_vm:
        result = next_vm
    elif next_vm < now_vm:
        result = max(next_vm, vm_min)
    else:
        result = now_vm
        
    return result


def check(now_vm, network):
    new_vm = now_vm
    
    if network//network_max > now_vm:
        new_vm = planning(network, now_vm)
    return new_vm
    


def prediction(now_vm):
    memories = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    networks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    result = np.array([[m, n] for m, n in zip(memories, networks)])

    dataX_reshaped = result.reshape(-1, 2)
    dataX_scaled = scalerX.fit_transform(dataX_reshaped)
    target = dataX_scaled.reshape(result.shape)

    target = torch.FloatTensor(target)
    target = target.reshape([1, seq_length, data_dim])

    out = model(target)
    pre = torch.flatten(out).item()

    pre = round(pre, 8)
    pre_inverse = scalerY.inverse_transform(np.array(pre).reshape(-1,1))
    result = pre_inverse.reshape([1])[0]


    planning(result, now_vm)

def predict(test_data, test_label):

    testX_tensor = torch.FloatTensor(test_data)
    testY_tensor = torch.FloatTensor(test_label)
    pred_now_vm = vm_min
    true_now_vm = vm_min
    with torch.no_grad():
        pred = []
        pred_vm = []
        true_vm = []
        fixed_vm = []
        temp = 0
        for pr in range(len(testX_tensor)):
            temp+=1
            
            model.reset_hidden_state()

            predicted = model(torch.unsqueeze(testX_tensor[pr], 0))
            predicted = torch.flatten(predicted).item()
            
            pred.append(predicted)

            pred_inverse = scalerY.inverse_transform(np.array(predicted).reshape(-1, 1))
            testY_inverse = scalerY.inverse_transform(testY_tensor[pr].reshape(-1, 1))
            
            pred_now_vm = planning(pred_inverse,pred_now_vm)
            true_now_vm = planning(testY_inverse,true_now_vm)

            pred_vm.append(pred_now_vm)
            true_vm.append(true_now_vm)

            if temp == 5:
                fixed_vm.append(pred_now_vm)
                fixed_now_vm = pred_now_vm
                temp = 0
            else:
                fixed_now_vm = check(fixed_now_vm, testY_inverse)
                fixed_vm.append(fixed_now_vm)


if __name__ == "__main__":
    prediction(vm_min)
