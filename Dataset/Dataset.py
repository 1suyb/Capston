import glob
from PIL import Image
from torchvision import transforms
from  torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset
import numpy as np


mean = np.array([0, 0, 0])
std = np.array([1, 1, 1])


class DVI2KDataset(Dataset) :
    def __init__(self, path, upscale_factor=4, train=True):
        # self.HR_img = []
        # self.LR_img = []
        if train :
            HRPath = path + r'\train\HR'
            if upscale_factor == 2 :
                LRPath = path + r'\train\LR\X2'
            elif upscale_factor == 4 :
                LRPath = path + r'\train\LR\X4'
            else :
                print("You input Invalid value in upscale_factor")
                return
        else :
            HRPath = path + r'\test\HR'
            if upscale_factor == 2 :
                LRPath = path + r'\test\LR\X2'
            elif upscale_factor == 4 :
                LRPath = path + r'\test\LR\X4'
            else :
                print("You input Invalid value in upscale_factor")
                return
        self.hr_img_list = glob.glob(HRPath + r'\*.png')
        self.lr_img_list = glob.glob(LRPath + r'\*.png')
        self.trans_lr = None
        self.trans_hr = transforms.Compose([transforms.Resize((128,128)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean,std)])
        if upscale_factor == 2 :
            self.trans_lr = transforms.Compose([transforms.Resize((64, 64)),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean,std)])
        elif upscale_factor == 4 :
            self.trans_lr = transforms.Compose([transforms.Resize((32, 32)),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean,std)])

    def __getitem__(self, index):
        img_hr = Image.open(self.hr_img_list[index])
        img_lr = Image.open(self.lr_img_list[index])
        hr = self.trans_hr(img_hr)
        lr = self.trans_lr(img_lr)
        return {"lr": lr, "hr": hr}

    def __len__(self):
        return len(self.hr_img_list)

class CropedDataset(Dataset) :
    def __init__(self, path, upscale_factor=4):
        self.img_list = glob.glob(path+r'\*.png')
        self.trans_hr = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean, std)])
        if upscale_factor ==4 :
            self.trans_lr = transforms.Compose([transforms.Resize((64, 64),
                                                                  interpolation=InterpolationMode.BILINEAR),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean, std)])
        elif upscale_factor == 2 :
            self.trans_lr = transforms.Compose([transforms.Resize((128, 128),
                                                                  interpolation=InterpolationMode.BILINEAR),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean, std)])
        else :
            print("잘못된 upscalefactor")
            return

    def __getitem__(self, index):
        img = Image.open(self.img_list[index])
        hr = self.trans_hr(img)
        lr = self.trans_lr(img)
        return {'lr' : lr, 'hr' : hr}

    def __len__(self):
        return len(self.img_list)


