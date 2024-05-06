import torch
import torch.nn as nn
from math import sqrt

class Model(nn.Module):
   def __init__(self):
       super().__init__()
       self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=2, padding=2)
       self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
       self.conv2 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=5, stride=2, padding=2)
       self.conv3 = nn.Conv2d(in_channels=64, out_channels=8, kernel_size=5, stride=2, padding=2)
       self.fc_1 = nn.Linear(in_features=288, out_features=29)
       self.init_weights()

   def init_weights(self):
       for conv in [self.conv1, self.conv2, self.conv3]:
           C_in = conv.weight.size(1)
           nn.init.normal_(conv.weight, 0.0, 1 / sqrt(5 * 5 * C_in))
           nn.init.constant_(conv.bias, 0.0)
       nn.init.normal_(self.fc_1.weight, 0.0, 1 / sqrt(32))
       nn.init.constant_(self.fc_1.bias, 0.0)

   def forward(self, x):
       x = self.conv1(x)
       x = torch.nn.ReLU()(x)
       x = self.pool(x)
       x = self.conv2(x)
       x = torch.nn.ReLU()(x)
       x = self.pool(x)
       x = self.conv3(x)
       x = torch.nn.ReLU()(x)
       x = x.view(x.size(0), -1)
       x = self.fc_1(x)
       return x
