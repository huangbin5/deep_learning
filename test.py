import numpy as np
import torch
from torch import nn
from torchvision import io

a = torch.tensor([[1, 2, 3], [4, 5, 6]])
b = torch.tensor([[1], [2]])
cross_entropy = nn.BCEWithLogitsLoss(reduction='sum')
print(cross_entropy(a, b))
