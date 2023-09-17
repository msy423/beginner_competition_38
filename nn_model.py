import numpy as np
import pandas as pd

import torch
from torchvision import transforms, models, datasets
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score




class beginer_mdoel(nn.Module):
    def __init__(self):
        super().__init__()
        self.leakyrelu=nn.LeakyReLU(negative_slope=0.01)
        self.liner1 = nn.Linear(11, 128)
        self.liner2 = nn.Linear(128, 64)
        self.liner3 = nn.Linear(64, 32)
        self.liner4 = nn.Linear(32, 16)
        self.liner5 = nn.Linear(16, 1)


    def forward(self, x):
        x = self.leakyrelu(self.liner1(x))
        x = self.leakyrelu(self.liner2(x))
        x = self.leakyrelu(self.liner3(x))
        x = self.leakyrelu(self.liner4(x))
        x = nn.Sigmoid(self.liner5(x))

        return x
