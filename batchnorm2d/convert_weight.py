"""
Convert pytorch resnet18 model.bn1 weight to numpy files
"""

import os

import numpy as np
from torchvision import models

if __name__ == "__main__":

    save_dir = "./weights"
    os.makedirs(save_dir, exist_ok=True)

    model = models.resnet18(weights = models.ResNet18_Weights.IMAGENET1K_V1)
    model.eval()

    bn1 = model.bn1
    weights = bn1.state_dict()

    for name in weights.keys():
        value = weights[name].detach().cpu().numpy()
        value = np.ascontiguousarray(value)
        value.tofile(os.path.join(save_dir, name + '.bin'))