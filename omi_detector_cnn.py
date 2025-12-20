import torch
import torch.nn as nn

class OMIDetector(nn.Module):
    def __init__(self):
        super(OMIDetector, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(12, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        self.fc = nn.Linear(64 * 250, 1) # Example dimensions

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return torch.sigmoid(self.fc(x))

if __name__ == "__main__":
    model = OMIDetector()
    print("OMI ECG Detector initialized.")
