
# Unlabeled Image Autoencoder

## Overview

This project is focused on building and utilizing an autoencoder for the unlabeled image datasets. The autoencoder is designed to compress the images into a lower-dimensional representation and then reconstruct them from this compressed form. This project includes the training of the autoencoder, extracting features from the any image test dataset, and visualizing the embeddings using t-SNE to reduce dimensionality further for visualization.

## Features

- **Autoencoder Model**: Utilizes a convolutional neural network (CNN) architecture for both the encoder and decoder.
- **Feature Extraction**: Extracts features from the any image test dataset using the trained encoder.
- **Data Visualization**: Applies t-SNE for dimensionality reduction on the extracted features and visualizes them, highlighting the ability to cluster and differentiate between different digits based on their features.

## Requirements

- Python 3.x
- PyTorch
- NumPy
- Matplotlib
- scikit-learn

## Installation

1. Clone the repository to your local machine.
2. Ensure you have Python 3.x installed.
3. Install the required packages using pip:

```bash
pip install torch torchvision numpy matplotlib scikit-learn
```

## Usage

1. **Training the Autoencoder**: Run the training script to train the autoencoder model on the MNIST dataset. The model will be saved in the `models` directory.

2. **Feature Extraction**: With the trained model, run the feature extraction script to process the MNIST test dataset through the encoder. The extracted features will be saved as `.npy` and `.csv` files.

3. **Data Visualization**: Use the t-SNE visualization script to read the extracted features and perform dimensionality reduction. The script will plot the 2D visualization of the MNIST digits based on their extracted features.

## Structure

- `model.py`: Defines the autoencoder model architecture.
- Training script: Includes data loading, model instantiation, training loop, and loss plotting.
- Feature extraction script: Loads the trained model and extracts features from the MNIST test dataset.
- t-SNE visualization script: Performs t-SNE on the extracted features and plots the results.

## Contributing

Contributions to this project are welcome. Please ensure that your code adheres to the project's coding standards and include appropriate unit tests where applicable.

## License

This project is licensed under the MIT License - see the LICENSE file for details.