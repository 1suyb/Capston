import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19
from torch.autograd import Function

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:18])

    def forward(self, img):
        return self.feature_extractor(img)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
        )
    def forward(self, x):
        return x + self.conv_block(x)

class Generator(nn.Module) :
    def __init__(self,in_channels=3, out_channels=3, n_residual_blocs=16, upscale_factor=4) :
        super(Generator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=64,kernel_size=3,stride=1,padding=1),
            nn.PReLU()
        )
        res_block = []
        for _ in range(n_residual_blocs) :
            res_block.append(ResidualBlock(64))
        self.res_block = nn.Sequential(*res_block)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64),
        )
        upsampling = []
        for out_feature in range(2) :
            upsampling += [
                nn.Conv2d(64,256,kernel_size=3,stride=1,padding=1),
                nn.PixelShuffle(upscale_factor=2),
                nn.PReLU()
            ]
        self.upsampling = nn.Sequential(*upsampling)
        self.conv3 = nn.Sequential(nn.Conv2d(64,out_channels=in_channels, kernel_size=9,stride=1,padding=4))

    def forward(self, x):
        out1 = self.conv1(x)
        out = self.res_block(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        out = self.upsampling(out)
        out = self.conv3(out)
        return out

class Discriminator(nn.Module) :
    def __init__(self,input_shape):
        super(Discriminator,self).__init__()

        self.input_shape = input_shape
        in_channels, in_height, in_width = self.input_shape
        patch_h, patch_w = int(in_height/2 **4),int(in_width/2**4)
        self.output_shape = (1, patch_h, patch_w)

        def discriminator_block(in_filters, outfilters, first_block=False) :
            layers = []
            layers.append(nn.Conv2d(in_filters,outfilters,kernel_size=3,stride=1,padding=1))
            if not first_block :
                layers.append(nn.BatchNorm2d(outfilters))
            layers.append(nn.LeakyReLU(0.2,inplace=True))
            layers.append(nn.Conv2d(outfilters,outfilters,kernel_size=3,stride=2,padding=1))
            layers.append(nn.BatchNorm2d(outfilters))
            layers.append(nn.LeakyReLU(0.2,inplace=True))
            return layers

        layers = []
        in_filters = in_channels
        for i,out_filters in enumerate([64,128,256,512]) :
            layers.extend(discriminator_block(in_filters,out_filters,first_block=(i==0)))
            in_filters=out_filters
        layers.append(nn.Conv2d(512,1,kernel_size=3,stride=1,padding=1))
        self.model = nn.Sequential(*layers)

    def forward(self,img):
        return self.model(img)

class Generator_bicubic(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_residual_blocs=16, upscale_factor=4):
        super(Generator_bicubic, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.PReLU()
        )
        res_block = []
        for _ in range(n_residual_blocs):
            res_block.append(ResidualBlock(64))
        self.res_block = nn.Sequential(*res_block)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
        )
        upsampling = []
        for out_feature in range(2):
            upsampling += [
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.Upsample(scale_factor=2, mode='bicubic'),
                nn.PReLU()
            ]
        self.upsampling = nn.Sequential(*upsampling)
        self.conv3 = nn.Sequential(nn.Conv2d(64, out_channels=out_channels, kernel_size=9, stride=1, padding=4))

    def forward(self, x):
        out1 = self.conv1(x)
        out = self.res_block(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        out = self.upsampling(out)
        out = self.conv3(out)
        return out

