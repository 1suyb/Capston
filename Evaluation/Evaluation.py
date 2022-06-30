import torch
import numpy as np
import cv2
from skimage.metrics._structural_similarity import  structural_similarity as ssim
# input image : tensor 1*c*h*w
# use image : numpy h*w*c

IMAGENET_MEAN, IMAGENET_STD = np.array([0.485,0.456,0.406]), np.array([0.229,0.224,0.225])

def Denormalize(img, IMAGENET=False) :
    if IMAGENET :
        img = np.clip(255.0 * (img * IMAGENET_STD + IMAGENET_MEAN), 0, 255)
    else :
        img = img * 255.0
    return img


def Tensor2NumpyImg(img, cuda=False, Normalize=True, IMAGENET=False) :
    img.squeeze()
    if cuda :
        img = img.detach.cuda().numpy()
    else :
        img = img.detach.cpu().numpy()

    if Normalize :
        img = Denormalize(img, IMAGENET)

    return img


def SSIM(img1, img2, Normalize=True, IMAGENET=False):
    img1 = Tensor2NumpyImg(img1)
    img2 = Tensor2NumpyImg(img2)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    (score, diff) = ssim(img1,img2,full=True)
    return score

def PSNR(self):
