import torch
import torch.nn as nn
from torchvision import models

class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        features = models.vgg16(pretrained=True).features
        self.to_relu_1_2 = nn.Sequential() 
        self.to_relu_2_2 = nn.Sequential() 
        self.to_relu_3_3 = nn.Sequential()
        self.to_relu_4_3 = nn.Sequential()

        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])
        for x in range(16, 23):
            self.to_relu_4_3.add_module(str(x), features[x])
        
        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.to_relu_1_2(x)
        h_relu_1_2 = h
        h = self.to_relu_2_2(h)
        h_relu_2_2 = h
        h = self.to_relu_3_3(h)
        h_relu_3_3 = h
        h = self.to_relu_4_3(h)
        h_relu_4_3 = h
        out = (h_relu_1_2, h_relu_2_2, h_relu_3_3, h_relu_4_3)
        return out


class Vgg16_2(nn.Module):
    def __init__(self):
        super(Vgg16_2, self).__init__()
        features = models.vgg16(pretrained=True).features
        self.to_relu_1_1 = nn.Sequential() 
        self.to_relu_2_1 = nn.Sequential() 
        self.to_relu_3_1 = nn.Sequential()
        self.to_relu_4_1 = nn.Sequential()
        self.to_relu_5_1 = nn.Sequential()

        for x in range(2):
            self.to_relu_1_1.add_module(str(x), features[x])
        for x in range(2, 7):
            self.to_relu_2_1.add_module(str(x), features[x])
        for x in range(7, 12):
            self.to_relu_3_1.add_module(str(x), features[x])
        for x in range(12, 19):
            self.to_relu_4_1.add_module(str(x), features[x])
        for x in range(19, 26):
            self.to_relu_5_1.add_module(str(x), features[x])
        
        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.to_relu_1_1(x)
        h_relu_1_1 = h
        h = self.to_relu_2_1(h)
        h_relu_2_1 = h
        h = self.to_relu_3_1(h)
        h_relu_3_1 = h
        h = self.to_relu_4_1(h)
        h_relu_4_1 = h
        h = self.to_relu_5_1(h)
        h_relu_5_1 = h
        out = (h_relu_1_1, h_relu_2_1, h_relu_3_1, h_relu_4_1, h_relu_5_1)
        return out

vgg = Vgg16()
print(vgg)