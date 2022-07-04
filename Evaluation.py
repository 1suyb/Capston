import math
from operator import mod
from tkinter.messagebox import showinfo
import torch
import numpy as np
import cv2
from skimage.metrics import  structural_similarity as ssim
from torchvision import transforms
from Model.SRGAN import Generator, Generator_bicubic, Discriminator, FeatureExtractor
from Model.SRGANSOCA import Generator_soca, Generator_bicubic_soca
from PIL import Image
import torch.nn as nn

# input image : tensor 1*c*h*w
# use image : numpy h*w*c


mean = np.array([0, 0, 0])
std = np.array([1, 1, 1])


ModelName = ["SRGAN", "SRGAN_bi", "SRGAN_SOCA", "SRGAN_SOCA_bi"]

IMAGENET_MEAN, IMAGENET_STD = np.array([0.485,0.456,0.406]), np.array([0.229,0.224,0.225])



def Tensor2NumpyImg(img, cuda=False) :
    img = img.squeeze()
    if cuda :
        img = img.detach().cuda().numpy()
    else :
        img = img.detach().cpu().numpy()
    
    img = np.transpose(img,(1,2,0))
    return img

def SSIM(img1, img2):
    img1 = Tensor2NumpyImg(img1)
    img2 = Tensor2NumpyImg(img2)
    img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    return ssim(img1,img2,full=True)

def PSNR(img1, img2):
    img1 = Tensor2NumpyImg(img1)
    img2 = Tensor2NumpyImg(img2)
    img1 = 255*img1.astype(np.float64)
    img2 = 255*img2.astype(np.float64)
    mse = np.mean((img1-img2)**2)
    if mse == 0 :
        return float('inf')
    return 20*math.log10(255/math.sqrt(mse))

def showimg(img) :
    cv2.imshow('img',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

class Evaluation : 
    def __init__(self,ImagePath=".\Data\Test\91.png") :
        img = Image.open(ImagePath)
        HR_tr = transforms.Compose([transforms.Resize((256, 256)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean, std)])
                        
        LR_tr = transforms.Compose([transforms.Resize((64, 64)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean, std)])
        
        HR = HR_tr(img)
        self.HR = HR.unsqueeze(0)
        LR = LR_tr(img)
        self.LR = LR.unsqueeze(0)

    def GetScore(self, model,savename) :
        SR = model(self.LR)
        psnr = PSNR(self.HR,SR)
        ssim = SSIM(self.HR,SR)

        cv2.imwrite("./Evaluation"+f"/{savename}.png",cv2.cvtColor(Tensor2NumpyImg(SR)*255, cv2.COLOR_BGR2RGB))
        cv2.imwrite("./Evaluation"+f"/{savename}_SSIMMap.png",ssim[1]*255)
        return (psnr,ssim)

    def sameimg(self) :
        return (PSNR(self.HR, self.HR), SSIM(self.HR, self.HR)[0])

    def Bilinear(self) :
        model = nn.Upsample(scale_factor=4,mode="bicubic")
        return self.GetScore(model,"bilinear")
    
    def SRGAN(self) :
        model = Generator()
        model.eval()
        model.load_state_dict(torch.load(f"./saved_models/{ModelName[0]}/generator.pth"))
        return self.GetScore(model,ModelName[0])
    
    def SRGAN_bi(self) :
        model = Generator_bicubic()
        model.eval()
        model.load_state_dict(torch.load(f"./saved_models/{ModelName[1]}/generator.pth"))
        return self.GetScore(model,ModelName[1])
    
    def SRGAN_SOCA(self) :
        model = Generator_soca()
        model.eval()
        model.load_state_dict(torch.load(f"./saved_models/{ModelName[2]}/generator.pth"))
        return self.GetScore(model,ModelName[2])

    def SRGAN_SOCA_bi(self) :
        model = Generator_bicubic_soca()
        model.eval()
        model.load_state_dict(torch.load(f"./saved_models/{ModelName[3]}/generator.pth"))
        return self.GetScore(model,ModelName[3])

    def SRGAN_ND(self) :
        model = Generator()
        model.eval()
        model.load_state_dict(torch.load(f"./saved_models/{ModelName[0]}_ND/generator.pth"))
        return self.GetScore(model,ModelName[0]+"_ND")
    
    def SRGAN_bi_ND(self) :
        model = Generator_bicubic()
        model.eval()
        model.load_state_dict(torch.load(f"./saved_models/{ModelName[1]}_ND/generator.pth"))
        return self.GetScore(model,ModelName[1]+"_ND")
    
    def SRGAN_SOCA_ND(self) :
        model = Generator_soca()
        model.eval()
        model.load_state_dict(torch.load(f"./saved_models/{ModelName[2]}_ND/generator.pth"))
        return self.GetScore(model,ModelName[2]+"_ND")

    def SRGAN_SOCA_bi_ND(self) :
        model = Generator_bicubic_soca()
        model.eval()
        model.load_state_dict(torch.load(f"./saved_models/{ModelName[3]}_ND/generator.pth"))
        return self.GetScore(model,ModelName[3]+"_ND")



if __name__ == "__main__" :
    eval = Evaluation(ImagePath=r"./Data/Test/1.png")
    print("same================================")
    print(eval.sameimg())

    bilnear = eval.Bilinear()
    print("Bilnear=============================")
    print(bilnear[0],bilnear[1][0])

    srgan = eval.SRGAN()
    print("SRGAN===============================")
    print(srgan[0],srgan[1][0])

    srgan_bi = eval.SRGAN_bi()
    print("SRGAN_bi============================")
    print(srgan_bi[0],srgan_bi[1][0])

    srgan_SOCA = eval.SRGAN_SOCA()
    print("SRGAN_SOCA===========================")
    print(srgan_SOCA[0],srgan_SOCA[1][0])

    srgan_SOCA_bi = eval.SRGAN_SOCA_bi()
    print("SRGAN_SOCA_bi========================")
    print(srgan_SOCA_bi[0],srgan_SOCA_bi[1][0])

    srgan_ND = eval.SRGAN_ND()
    print("SRGAN_ND===============================")
    print(srgan_ND[0],srgan_ND[1][0])

    srgan_bi_ND = eval.SRGAN_bi_ND()
    print("SRGAN_bi_ND============================")
    print(srgan_bi_ND[0],srgan_bi_ND[1][0])

    srgan_SOCA_ND = eval.SRGAN_SOCA_ND()
    print("SRGAN_SOCA_ND===========================")
    print(srgan_SOCA_ND[0],srgan_SOCA_ND[1][0])

    srgan_SOCA_bi_ND = eval.SRGAN_SOCA_bi_ND()
    print("SRGAN_SOCA_bi_ND========================")
    print(srgan_SOCA_bi_ND[0],srgan_SOCA_bi_ND[1][0])
