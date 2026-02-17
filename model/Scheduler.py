import numpy as np
import numpy.random as npr
import torch
import torch.nn as nn
import torch.optim as optim
import copy

# Link to TD3 model
# https://github.com/sfujim/TD3/blob/master/TD3.py


class Scheduler(nn.Module):
    def __init__(self, state_dim, hidden_dim=256, input_dim=256, cuda="0"):
        super(Scheduler, self).__init__()
        self.device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
        self.layer1 = nn.Linear(state_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, input_dim)
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, device=self.device)
        self.layer4 = nn.Linear(state_dim, hidden_dim)
        self.layer5 = nn.Linear(hidden_dim, hidden_dim)
        self.layer6 = nn.Linear(hidden_dim, input_dim)
        self.to(self.device)

    def forward(self, x):
        # encoder
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        # lstm layer
        x, (hn, cn) = self.lstm(x)
        # decoder
        x = torch.relu(self.layer4(x))
        x = torch.relu(self.layer5(x))
        x = torch.sigmoid(self.layer6(x))
        return x
