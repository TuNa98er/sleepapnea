import torch
import torch.nn as nn

class LeNet5(nn.Module):
    def __init__(self, num_classes):
        super(LeNet5, self).__init__()
        
      
        self.layer1 = nn.Sequential(
            nn.Conv1d(3, 16, kernel_size=7, stride=2, padding=0),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size = 3, stride = 3))
        
        self.layer2 = nn.Sequential(
            nn.Conv1d(16, 128, kernel_size=7, stride=2, padding=0),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size = 3, stride = 3))
        
        self.layer3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=7, stride=2, padding=0),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size = 3, stride = 3)
            )
        self.drop=nn.Dropout(0.4)
        self.fc = nn.Linear(768, 512)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(512, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # out = self.layer(x))
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.drop(out)
        # out = self.layer4(out)
        
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out
