import torch 
import torch.nn as nn

class FFNet(nn.Module):
    def __init__(self, features, hidden_1, hidden_2, num_classes, p_dropout):
        super(FFNet, self).__init__()
        self.input = nn.Linear(features, hidden_1)
        self.fc_long = nn.Linear(hidden_1, hidden_1)
        self.fc_long_short = nn.Linear(hidden_1, hidden_2)
        self.fc_short = nn.Linear(hidden_2, hidden_2)
        self.output = nn.Linear(hidden_2, num_classes)

        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p_dropout)
        #self.act_output = nn.Sigmoid()
        self.bn_long  = nn.BatchNorm1d(hidden_1)
        self.bn_short = nn.BatchNorm1d(hidden_2)
        self.act_output = nn.Softmax()

    def forward(self, x):
        #input
        x = self.input(x)

        #block 1
        x = self.bn_long(self.fc_long(x))
        x = self.relu(x)
        x = self.drop(x)

        ##block 2
        x = self.bn_long(self.fc_long(x))
        x = self.relu(x)
        x = self.drop(x)
        
        #block 3
        x = self.bn_short(self.fc_long_short(x))
        x = self.relu(x)
        x = self.drop(x)

        ##block 4
        x = self.bn_short(self.fc_short(x))
        x = self.relu(x)
        x = self.drop(x)

        #output
        x = self.output(x)
        x = self.act_output(x)

        return x
