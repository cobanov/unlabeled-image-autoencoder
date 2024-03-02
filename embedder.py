import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

test_dataset = datasets.MNIST("./data", train=False, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load("models/model_9.pth").to(device)


with torch.no_grad():
    features = []
    for i, (data, _) in enumerate(test_loader):
        data = data.to(device)
        encoded = model.encoder(data)
        features.append(encoded.view(encoded.size(0), -1).cpu().numpy())

features = np.concatenate(features, axis=0)
print(features.shape)

# Delete rows with all zeros
features = features[~np.all(features == 0, axis=1)]
print(f"After deleting all zeros: {features.shape}")

# Save embeddings as a numpy array
np.save("embeddings.npy", features)
np.savetxt("embeddings.csv", features, delimiter=",")
