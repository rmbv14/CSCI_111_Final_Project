import pandas as pd
import torch
import numpy as np
from torch import nn
from sklearn.model_selection import train_test_split

# Load and prepare data
df = pd.read_csv('diabetes_data.csv')
df.dropna(inplace=True)

# Split data
x = df.drop(columns=['Outcome'])
y = df['Outcome']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

# Convert to tensors
x_train_tensor = torch.tensor(x_train.values).to(torch.float32)
x_test_tensor = torch.tensor(x_test.values).to(torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# Neural Network model definition
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        #input layer
        self.layer1 = nn.Linear(8,70)
        self.relu1 = nn.ReLU()

        self.layer2 = nn.Linear(70,70)
        self.relu2 = nn.ReLU()

        self.layer3 = nn.Linear(70,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu1(x)

        x = self.layer2(x)
        x = self.relu2(x)

        x = self.layer3(x)
        pred = self.sigmoid(x)
        return pred 