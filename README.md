# Neural Net

Created a 2-layer neural network that can classify digits in the
[MNIST dataset](https://en.wikipedia.org/wiki/MNIST_database)
with 98.1% accuracy with the pretrained model (data/pretrained.model).

The model was trained with 800 hidden layers with a 5% learning
rate and 20 epochs.

Potential future additions would be SIMD/GPU acceleration as the
model does take some time to train, ~30 min for the model. Smaller
models can be trained with ~97% in less than a minute however.

## Dataset

The dataset can be downloaded on
[Kaggle](https://www.kaggle.com/oddrationale/mnist-in-csv).