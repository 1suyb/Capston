import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19
from torch.autograd import Function
from Model.MPNCOV import MPNCOV

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:18])

    def forward(self, img):
        return self.feature_extractor(img)

class ResidualBlock_Soca(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock_Soca, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
        )
        self.conv2 = SOCA(64)

    def forward(self, x):
        out = self.conv_block(x)
        out = self.conv2(out)
        return x + out

class Generator_soca(nn.Module) :
    def __init__(self,in_channels=3, out_channels=3, n_residual_blocs=16, upscale_factor=4) :
        super(Generator_soca, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=64,kernel_size=3,stride=1,padding=1),
            nn.PReLU()
        )
        res_block = []
        for _ in range(n_residual_blocs) :
            res_block.append(ResidualBlock_Soca(64))
        self.res_block = nn.Sequential(*res_block)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64,0.8),
        )
        upsampling = []
        for out_feature in range(2) :
            upsampling += [
                nn.Conv2d(64,256,kernel_size=3,stride=1,padding=1),
                nn.PixelShuffle(upscale_factor=2),
                nn.PReLU()
            ]
        self.upsampling = nn.Sequential(*upsampling)
        self.conv3 = nn.Sequential(nn.Conv2d(64,out_channels=out_channels,kernel_size=9,stride=1,padding=4))

    def forward(self, x):
        out1 = self.conv1(x)
        out = self.res_block(out1)
        out2 = self.conv2(out)
        out = torch.add(out1,out2)
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


class Generator_bicubic_soca(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_residual_blocs=8, upscale_factor=4):
        super(Generator_bicubic_soca, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.PReLU()
        )
        res_block = []
        for _ in range(n_residual_blocs):
            res_block.append(ResidualBlock_Soca(64))
        self.res_block = nn.Sequential(*res_block)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),

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

class SOCA(nn.Module) :
    def __init__(self, in_channels, reduction=8):
        super(SOCA, self).__init__()

        #self.max_pool = nn.MaxPool2d(kernel_size=2)
        self.conv_du = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//reduction, in_channels, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
    def forward(self,x):
        batch_size, C, h, w = x.shape
        N = int(h*w)
        min_h = min(h,w)
        h1 = 1000
        w1 = 1000
        if h<h1 and w<w1:
            x_sub=x
        elif h<h1 and w > w1 :
            W = (w-w1)//2
            x_sub = x[:,:,:,W:(W+w1)]

        elif w<w1 and h>h1 :
            H = (h-h1)//2
            x_sub = x[:,:,H:(H+h1),:]
        else :
            H = (h-h1)//2
            W = (w-w1)//2
            x_sub = x[:,:,H:(H+h1),W:(W+w1)]
        cov_mat = MPNCOV.CovpoolLayer(x_sub)
        cov_mat_sqrt = MPNCOV.SqrtmLayer(cov_mat,5)
        cov_mat_sum = torch.mean(cov_mat_sqrt,1)
        cov_mat_sum = cov_mat_sum.view(batch_size,C,1,1)
        y_cov = self.conv_du(cov_mat_sum)
        return y_cov*x

if __name__ == "__main__" :
    t = MPNCOV.Covpool()