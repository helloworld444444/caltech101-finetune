# model.py
import torch.nn as nn
from torchvision import models
from config import Config

def create_model(pretrained=True):
    model = models.resnet18(pretrained = pretrained if pretrained else None )
    model.fc = nn.Linear(model.fc.in_features, Config.num_classes)
    return model
