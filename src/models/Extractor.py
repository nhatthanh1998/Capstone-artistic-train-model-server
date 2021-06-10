import torch.nn as nn
from torchvision import models 

class Extractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.vgg16(pretrained=True).features[:23]
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), self.model[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), self.model[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), self.model[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), self.model[x])

            
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        features = {}
        x = self.slice1(x)
        features['relu1_2'] = x
        x = self.slice2(x)
        features['relu2_2']  = x
        x = self.slice3(x)
        features['relu3_3'] = x
        x = self.slice4(x)
        features['relu4_3'] = x
        
        return features
