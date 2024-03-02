import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import Autoencoder

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)
train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST("./data", train=False, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Autoencoder().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Training loop
epochs = 10
train_losses = []
val_losses = []

for epoch in range(epochs):
    train_loss = 0
    val_loss = 0

    # Training phase
    model.train()
    for i, (train_data, _) in enumerate(train_loader):
        train_data = train_data.to(device)
        optimizer.zero_grad()
        train_outputs = model(train_data)
        loss = criterion(train_outputs, train_data)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * train_data.size(0)

    # Validation phase
    model.eval()
    with torch.no_grad():
        for i, (val_data, _) in enumerate(test_loader):
            val_data = val_data.to(device)
            val_outputs = model(val_data)
            loss = criterion(val_outputs, val_data)
            val_loss += loss.item() * val_data.size(0)

    train_loss = train_loss / len(train_loader.dataset)
    val_loss = val_loss / len(test_loader.dataset)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    print(
        f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
    )

    # Save model state dictionary
    torch.save(model.state_dict(), f"models/model_{epoch}.pth")


# Plot the training and validation losses
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
