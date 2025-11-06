
import torch.nn as nn




















































        





import torch
import torch.nn as nn
import torch.nn.functional as F

class OneDCNN(nn.Module):
    def __init__(self, input_dim, hidden_size=64):
        super(OneDCNN, self).__init__()

        self.conv1 = nn.Conv1d(1, hidden_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size * 2, kernel_size=3, padding=1)

        self.dropout = nn.Dropout(0.8)  
        
        self.relu = nn.ReLU()

        
        self.pool = nn.AdaptiveAvgPool1d(hidden_size)  

        self.fc1 = nn.Linear(hidden_size * 2 * hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

        self.cnn_activations = None
        self.cnn_grads = None

    
    

    
    
    
    

    
    
    

    
    

    
    

    
    
    
    

    def forward(self, x, return_data="pred", goal="regression", alt="none"):
        x = x.float().unsqueeze(1)  

        x = self.relu(self.conv1(x))
        x = self.dropout(x)
        x = self.relu(self.conv2(x))
        x = self.dropout(x)

        if x.requires_grad:
            x.register_hook(self.save_cnn_grads)
        self.cnn_activations = x

        x = self.pool(x)                  
        x = x.view(x.size(0), -1)         

        x_rep = self.relu(self.fc1(x))
        output = self.fc2(x_rep)          

        pred = output.squeeze()  
        if return_data == "pred":
            return pred
        else:
            return pred, x_rep


    def save_cnn_grads(self, grad):
        self.cnn_grads = grad





class LSTM(nn.Module):
    def __init__(self, input_dim, bidirection, hidden_size=128, num_layers=2):
        super(LSTM, self).__init__()

        
        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=bidirection)  

        
        if bidirection:
            self.fc1 = nn.Linear(hidden_size * 2, hidden_size)  
        else:
            self.fc1 = nn.Linear(hidden_size, hidden_size)  
        self.fc2 = nn.Linear(hidden_size, 1)  

        
        self.dropout = nn.Dropout(0.3)

        
        self.relu = nn.ReLU()

    def forward(self, x, return_data = "pred"):
        
        x = x.float()
        if len(x.shape) == 2:  
            x = x.unsqueeze(1)  

        
        lstm_out, _ = self.lstm(x)  
        
        x = lstm_out[:, -1, :]  
        
        x = self.relu(self.fc1(x))  
        x_rep = self.dropout(x)  
        output = self.relu(self.fc2(x_rep))  
        if return_data == "pred":
            return output.squeeze()  
        else:
            return output.squeeze(), x_rep


import torch
import torch.nn as nn

class GRU(nn.Module):
    def __init__(self, input_dim, bidirection, hidden_size=128, num_layers=2):
        super(GRU, self).__init__()

        
        self.gru = nn.GRU(input_size=input_dim,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        batch_first=True,
                        bidirectional=bidirection)  

        
        if bidirection:
            self.fc1 = nn.Linear(hidden_size * 2, 128)  
        else:
            self.fc1 = nn.Linear(hidden_size, 128)  
        self.fc2 = nn.Linear(128, 1)  

        
        self.dropout = nn.Dropout(0.3)

        
        self.relu = nn.ReLU()

    def forward(self, x):
        
        x = x.float()
        if len(x.shape) == 2:  
            x = x.unsqueeze(1)  
        
        gru_out, _ = self.gru(x)  

        
        x = gru_out[:, -1, :]  

        
        x = self.relu(self.fc1(x))  
        x = self.dropout(x)  
        output = self.fc2(x)  

        return output.squeeze()  


