from Model.SRGAN import Generator, Generator_bicubic, Discriminator, FeatureExtractor
from Model.SRGANSOCA import Generator_soca, Generator_bicubic_soca
from Dataset import Dataset
import torch
from tqdm import tqdm
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image, make_grid
from tqdm import tqdm
import os

ModelName = ["SRGAN", "SRGAN_bi", "SRGAN_SOCA", "SRGAN_SOCA_bi"]


class Train:
    def __init__(self, load_pretrained=False,
                 dataset_path=r"\ImageDatas",
                 n_epoch=20,
                 batch_size=5,
                 lr=0.00008,
                 b1=0.5,
                 b2=0.999,
                 n_cpu=8,
                 hr_height=256,
                 hr_width=256,
                 channels=3,
                 model="SRGAN"):
        self.load_pretrained_model = load_pretrained
        self.dataset_path = dataset_path
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.n_cpu = n_cpu
        self.hr_shape = (hr_height, hr_width)
        self.channels = channels
        self.train_dataloader = DataLoader(Dataset.CropedDataset(dataset_path + r"\Train"),
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=n_cpu)
        self.test_dataloader = DataLoader(Dataset.CropedDataset(dataset_path + r"\Test"),
                                          batch_size=1,
                                          num_workers=n_cpu)
        self.model = model
        os.makedirs(f".\saved_models\{self.model}",exist_ok=True)
        os.makedirs(f".\saved_models\{self.model}_ND",exist_ok=True)
        os.makedirs(f".\images\{self.model}",exist_ok=True)
        os.makedirs(f".\images\{self.model}_ND",exist_ok=True)

    
    def Train_D(self):
        cuda = True if torch.cuda.is_available() else False
        torch.manual_seed(777)
        if cuda:
            torch.cuda.manual_seed_all(777)
            torch.cuda.empty_cache()
        model_gen = 0
        if self.model == ModelName[0]:
            model_gen = Generator()
        elif self.model == ModelName[1]:
            model_gen = Generator_bicubic()
        elif self.model == ModelName[2]:
            model_gen = Generator_soca()
        elif self.model == ModelName[3]:
            model_gen = Generator_bicubic_soca()
        else:
            print("잘못된 모델 이름")
            exit(1)

        model_dis = Discriminator(input_shape=(self.channels, *self.hr_shape))

        feature_extractor = FeatureExtractor()

        criterion_gan = torch.nn.MSELoss()
        criterion_content = torch.nn.L1Loss()
        criterion_content_mse = torch.nn.MSELoss()

        if cuda:
            model_gen = model_gen.cuda()
            model_dis = model_dis.cuda()
            feature_extractor.cuda()
            criterion_gan = criterion_gan.cuda()
            criterion_content = criterion_content.cuda()
            criterion_content_mse = criterion_content_mse.cuda()

        if self.load_pretrained_model:
            model_gen.load_state_dict(torch.load(f"./saved_models/{self.model}/generator.pth"))
            model_dis.load_state_dict(torch.load(f"./saved_models/{self.model}/discriminator.pth"))

        optim_G = torch.optim.Adam(model_gen.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        optim_D = torch.optim.Adam(model_dis.parameters(), lr=self.lr, betas=(self.b1, self.b2))

        Tensor = torch.FloatTensor().cuda() if cuda else torch.FloatTensor()

        train_gen_losses, train_disc_losses, train_counter = [], [], []
        test_gen_losses, test_disc_losses = [], []
        test_counter = [idx * len(self.train_dataloader.dataset) for idx in range(1, self.n_epoch + 1)]

        for epoch in range(self.n_epoch):
            gen_loss, disc_loss = 0, 0
            tqdm_bar = tqdm(self.train_dataloader, desc=f'Training Epoch{epoch}',
                            total=int(len(self.train_dataloader)))
            for batch_idx, imgs in enumerate(tqdm_bar):
                model_gen.train()
                model_dis.train()
                imgs_lr = Variable(imgs['lr'].type(torch.Tensor)).cuda()
                imgs_hr = Variable(imgs["hr"].type(torch.Tensor)).cuda()
                valid = Variable(torch.Tensor(np.ones((imgs_lr.size(0), *model_dis.output_shape))),
                                    requires_grad=False).cuda()
                fake = Variable(torch.Tensor(np.ones((imgs_lr.size(0), *model_dis.output_shape))),
                                requires_grad=False).cuda()

                optim_G.zero_grad()
                imgs_sr = model_gen(imgs_lr)
                loss_GAN = criterion_gan(model_dis(imgs_sr), valid)
                gen_features = feature_extractor(imgs_sr)
                real_feature = feature_extractor(imgs_hr)
                loss_content_mse = criterion_content_mse(imgs_sr, imgs_hr)
                loss_content = criterion_content(gen_features, real_feature.detach()).cuda()
                loss_G = 0.7 * loss_content + loss_content_mse + 1e-3 * loss_GAN
                loss_G.backward()
                optim_G.step()

                optim_D.zero_grad()
                loss_real = criterion_gan(model_dis(imgs_hr), valid)
                loss_fake = criterion_gan(model_dis(imgs_sr.detach()), fake)
                loss_D = (loss_real + loss_fake) / 2
                loss_D.backward()
                optim_D.step()

                gen_loss += loss_G.item()
                train_gen_losses.append(loss_G.item())

                disc_loss += loss_D.item()
                train_disc_losses.append(loss_D.item())

                train_counter.append(
                    batch_idx * self.batch_size + imgs_lr.size(0) + epoch * len(self.train_dataloader.dataset))
                tqdm_bar.set_postfix(gen_loss=gen_loss / (batch_idx + 1), disc_loss=disc_loss / (batch_idx + 1))

            gen_loss, disc_loss = 0, 0
            tqdm_bar = tqdm(self.test_dataloader, desc=f'Testing Epoch{epoch}',
                            total=int(len(self.test_dataloader.dataset)))
            for batch_idx, imgs in enumerate(tqdm_bar):
                model_gen.eval()
                model_dis.eval()
                imgs_lr = Variable(imgs['lr'].type(torch.Tensor)).cuda()
                imgs_hr = Variable(imgs['hr'].type(torch.Tensor)).cuda()

                valid = Variable(torch.Tensor(np.ones((imgs_lr.size(0), *model_dis.output_shape))),
                                    requires_grad=False).cuda()
                fake = Variable(torch.Tensor(np.zeros((imgs_lr.size(0), *model_dis.output_shape))),
                                requires_grad=False).cuda()

                imgs_sr = model_gen(imgs_lr)
                loss_GAN = criterion_gan(model_dis(imgs_sr), valid)
                gen_features = feature_extractor(imgs_sr)
                real_feature = feature_extractor(imgs_hr)

                loss_content = criterion_content(gen_features, real_feature.detach()).cuda()
                loss_G = loss_content + 1e-3 * loss_GAN
                loss_real = criterion_gan(model_dis(imgs_hr), valid)
                loss_fake = criterion_gan(model_dis(imgs_sr.detach()), fake)
                loss_D = (loss_real + loss_fake) / 2
                gen_loss += loss_G.item()
                disc_loss += loss_D.item()
                tqdm_bar.set_postfix(gen_loss=gen_loss / (batch_idx + 1))
                if batch_idx == 1:
                    imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=4, mode='bilinear')
                    imgs_hr = make_grid(imgs_hr, nrow=1, normalize=True)
                    imgs_sr = make_grid(imgs_sr, nrow=1, normalize=True)
                    imgs_lr = make_grid(imgs_lr, nrow=1, normalize=True)
                    imgs_grid = torch.cat((imgs_hr, imgs_lr, imgs_sr), -1)
                    save_image(imgs_grid, f"./images/{self.model}/{epoch}.png", normalize=False)

            test_gen_losses.append(gen_loss / len(self.test_dataloader))
            test_disc_losses.append(disc_loss / len(self.test_dataloader))
            if np.argmin(test_gen_losses) == len(test_gen_losses) - 1:
                torch.save(model_gen.state_dict(), f"./saved_models/{self.model}/generator.pth")
                torch.save(model_dis.state_dict(), f"./saved_models/{self.model}/discriminator.pth")
    
    def Train_ND(self):
        cuda = True if torch.cuda.is_available() else False
        torch.manual_seed(777)
        if cuda:
            torch.cuda.manual_seed_all(777)
            torch.cuda.empty_cache()
        model_gen = 0
        if self.model == ModelName[0]:
            model_gen = Generator()
        elif self.model == ModelName[1]:
            model_gen = Generator_bicubic()
        elif self.model == ModelName[2]:
            model_gen = Generator_soca()
        elif self.model == ModelName[3]:
            model_gen = Generator_bicubic_soca()
        else:
            print("잘못된 모델 이름")
            exit(1)

        feature_extractor = FeatureExtractor()

        criterion_gan = torch.nn.MSELoss()
        criterion_content = torch.nn.L1Loss()
        criterion_content_mse = torch.nn.MSELoss()

        if cuda:
            model_gen = model_gen.cuda()
            feature_extractor.cuda()
            criterion_gan = criterion_gan.cuda()
            criterion_content = criterion_content.cuda()
            criterion_content_mse = criterion_content_mse.cuda()

        if self.load_pretrained_model:
            model_gen.load_state_dict(torch.load(f"./saved_models/{self.model}/generator.pth"))

        optim_G = torch.optim.Adam(model_gen.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        Tensor = torch.FloatTensor().cuda() if cuda else torch.FloatTensor()

        train_gen_losses, train_disc_losses, train_counter = [], [], []
        test_gen_losses, test_disc_losses = [], []
        test_counter = [idx * len(self.train_dataloader.dataset) for idx in range(1, self.n_epoch + 1)]

        for epoch in range(self.n_epoch):
            gen_loss = 0
            tqdm_bar = tqdm(self.train_dataloader, desc=f'Training Epoch{epoch}',
                            total=int(len(self.train_dataloader)))
            for batch_idx, imgs in enumerate(tqdm_bar):
                model_gen.train()
                imgs_lr = Variable(imgs['lr'].type(torch.Tensor)).cuda()
                imgs_hr = Variable(imgs["hr"].type(torch.Tensor)).cuda()

                optim_G.zero_grad()
                imgs_sr = model_gen(imgs_lr)
                gen_features = feature_extractor(imgs_sr)
                real_feature = feature_extractor(imgs_hr)

                loss_content = criterion_content(gen_features, real_feature.detach()).cuda()
                loss_G = loss_content
                loss_G.backward()
                optim_G.step()

                gen_loss += loss_G.item()
                train_gen_losses.append(loss_G.item())

                train_counter.append(
                    batch_idx * self.batch_size + imgs_lr.size(0) + epoch * len(self.train_dataloader.dataset))
                tqdm_bar.set_postfix(gen_loss=gen_loss / (batch_idx + 1))
            gen_loss, disc_loss = 0, 0
            tqdm_bar = tqdm(self.test_dataloader, desc=f'Testing Epoch{epoch}',
                            total=int(len(self.test_dataloader.dataset)))
            for batch_idx, imgs in enumerate(tqdm_bar):
                model_gen.eval()
                imgs_lr = Variable(imgs['lr'].type(torch.Tensor)).cuda()
                imgs_hr = Variable(imgs['hr'].type(torch.Tensor)).cuda()

                imgs_sr = model_gen(imgs_lr)
                gen_features = feature_extractor(imgs_sr)
                real_feature = feature_extractor(imgs_hr)

                loss_content = criterion_content(gen_features, real_feature.detach()).cuda()
                loss_G = loss_content
                gen_loss += loss_G.item()
                tqdm_bar.set_postfix(gen_loss=gen_loss / (batch_idx + 1))

                if batch_idx == 1:
                    imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=4, mode='bilinear')
                    imgs_hr = make_grid(imgs_hr, nrow=1, normalize=True)
                    imgs_sr = make_grid(imgs_sr, nrow=1, normalize=True)
                    imgs_lr = make_grid(imgs_lr, nrow=1, normalize=True)
                    imgs_grid = torch.cat((imgs_hr, imgs_lr, imgs_sr), -1)
                    save_image(imgs_grid, f"./images/{self.model}_ND/{epoch}.png", normalize=False)
                test_gen_losses.append(gen_loss / len(self.test_dataloader))
                test_disc_losses.append(disc_loss / len(self.test_dataloader))
                if np.argmin(test_gen_losses) == len(test_gen_losses) - 1:
                    torch.save(model_gen.state_dict(), f"./saved_models/{self.model}_ND/generator.pth")
