import torch
import torch.nn as nn

GRID = 7

class SimpleYOLO(nn.Module):
    
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 256 → 128

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 128 → 64

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 64 → 32
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 32 * 32, 1024),
            nn.ReLU(),
            nn.Linear(1024, GRID * GRID * 5)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        x = x.view(-1, GRID, GRID, 5)

        # 🔧 Stabilisierung
        x[..., 0:2] = torch.sigmoid(x[..., 0:2])   # x, y → [0,1]
        x[..., 2:4] = torch.exp(x[..., 2:4])       # w, h → positiv
        x[..., 4]   = torch.sigmoid(x[..., 4])     # confidence → [0,1]
        return x