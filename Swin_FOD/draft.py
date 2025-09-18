# %%
import time
import numpy as np
import torch
from torch import nn
from torch.nn import MSELoss, Dropout, Softmax, Linear, Conv2d, LayerNorm, GELU

import torch
import torchvision.models as models

from encoder import generate_model

# Load a pre-trained ResNet18 model
resnet = generate_model(model_depth=18, n_input_channels=1, n_classes=3, conv1_t_stride=2)

print(f"{list(resnet.children())} layers")

# Remove the final fully connected layer
encoder = torch.nn.Sequential(*list(resnet.children())[:-4])

# Example input
input_tensor = torch.randn(1, 1, 256, 256, 256) 

# Extract features
features = encoder(input_tensor)

print(features.shape)