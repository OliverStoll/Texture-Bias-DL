import torch

# Hyperparameters
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 2
layer_sizes = [input_size, 50, num_classes]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
