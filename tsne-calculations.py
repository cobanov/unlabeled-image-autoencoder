import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

test_dataset = datasets.MNIST("./data", train=False, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Load embeddings
features = np.load("embeddings.npy")
print(features.shape)

tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, init="pca")
tsne_features = tsne.fit_transform(features)

plt.figure(figsize=(10, 6))
plt.scatter(
    tsne_features[:, 0], tsne_features[:, 1], c=test_dataset.targets, cmap="viridis"
)
plt.xlabel("TSNE 1")
plt.ylabel("TSNE 2")
plt.colorbar(label="Class")
plt.show()
