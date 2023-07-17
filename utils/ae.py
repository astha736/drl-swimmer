import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np


class ObservationDataSet(Dataset):
    def __init__(self, dir):
        self.data = np.squeeze(np.load(dir))[:, :10]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class AE(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(10, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 5),
        )

        self.decoder = nn.Sequential(
            nn.Linear(5, 16),
            nn.Tanh(),
            nn.Linear(16, 32),
            nn.Tanh(),
            nn.Linear(32, 10),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


data = ObservationDataSet("203.npy")
# data = ObservationDataSet("1.npy")
loader = torch.utils.data.DataLoader(data, batch_size=256, shuffle=True)

# Model Initialization
model = AE()

# Validation using MSE Loss function
loss_function = torch.nn.MSELoss()

# Using an Adam Optimizer with lr = 0.1
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training
epochs = 200
losses = []
for epoch in range(epochs):
    for data in loader:
        data = data.to(torch.float32)
        out = model(data)

        loss = loss_function(out, data)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch: {epoch + 1}, Loss: {loss.item():.5f}")
