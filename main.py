import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class DoubleConv(nn.Module):
    def __init__(self, inchannels, outchannels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inchannels, outchannels,3,1),
            nn.Batchnorm2d(outchannels),
            nn.ReLU(inplace= True),
            nn.Conv2d(inchannels, outchannels,3,1),
            nn.Batchnorm2d(outchannels),
            nn.ReLU(inplace= True)
        )
    def forward(self,x):
        return self.conv(x)
    
class Unet(nn.Module):
    def __init__(
            self, inchannels =3, outchannels=1, features =[64,128,256,512]
                 ):
        super.__init__(Unet,self)
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size =2,stride =2)
        for feature in features:
            pass