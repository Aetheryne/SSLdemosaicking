# import libs
import numpy as np
import cv2
import os
import gc
import math

from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader, Dataset

import tqdm

# CFA sampling
def CFA(pic: np.ndarray):
    #h, w -> height,width(cv2-format)
    h, w, _ = pic.shape
    RGGB = np.array([[[1, 0, 0], [0, 1, 0]], [[0, 1, 0], [0, 0, 1]]])
    # multiples of template tiling
    time_h = int(np.ceil(h / 2))
    time_w = int(np.ceil(w / 2))
    # template tiling
    CFA = np.tile(RGGB, (time_h, time_w, 1))
    CFA = CFA[:h, :w, :]
    #CFA template sampling
    processed = pic * CFA
    return processed

# CFA filtering on 4D Tensor, the parameter "pattern" designates the pattern string(RGGB, BGGR, etc.)
def CFA_d4(pic: torch.Tensor, pattern: str):
    #b, c, h, w -> batch_size, channel, height, width
    b, c, h, w = pic.shape
    pic = pic.cuda()
    
    processed = torch.zeros(pic.shape)
    processed = processed.cuda()
    
    RGGB = np.array([[[1, 0, 0], [0, 1, 0]], [[0, 1, 0], [0, 0, 1]]])
    
    #multiples of template tiling
    time_h = int(np.ceil(h / 2))
    time_w = int(np.ceil(w / 2))
    
    #tiled RGGB
    CFA = np.tile(RGGB, (time_h, time_w, 1))
    processed2 = torch.clone(pic)
    processed2 = processed2.cuda()
   
    CFA = CFA[:h, :w, :]
    #CFA -> h*w*3,RGB
    CFA3 = CFA.transpose((2,0,1))
    #CFA3 -> 3*h*w,RGB
    # numpy -> tensor
    CFA3 = torch.from_numpy(CFA3)
    CFA3 = CFA3.cuda()
    
    for i in range(b):
        if pattern == "BGGR":
            #translate BGGR -> RGGB
            processed2[i] = torch.roll(processed2[i],shifts=(-1, -1), dims = (1,2))
        elif pattern == "GBRG":
            #translate GBRG -> RGGB
            processed2[i] = torch.roll(processed2[i],shifts=(-1, 0), dims = (1,2))
        elif pattern == "GRBG":
            #translate GRBG -> RGGB
            processed2[i] = torch.roll(processed2[i],shifts=(0, -1), dims = (1,2))
    for i in range(b):
        processed2[i] = processed2[i] * CFA3
        
    return processed2

# "img_o" is the input and "fil" is the convolution kernel to perform convolution for bilinear interpolation
def my_fil(pic):
    h, w, _ = pic.shape
    # RGB -> BGR, to split
    pic = pic[:,:,::-1]
    # resolving memory discontinuity problems
    pic = pic.copy()
    [B, G, R] = cv2.split(pic)
    filter = np.array([[[0.25, 0., 0.25],
                        [0.5, 0.25, 0.5],
                        [0.25, 0., 0.25]],

                       [[0.5, 0.25, 0.5],
                        [1., 1., 1.],
                        [0.5, 0.25, 0.5]],

                       [[0.25, 0., 0.25],
                        [0.5, 0.25, 0.5],
                        [0.25, 0., 0.25]]])

    B_fil = cv2.filter2D(B, -1, kernel=filter[:, :, 2])
    G_fil = cv2.filter2D(G, -1, kernel=filter[:, :, 1])
    R_fil = cv2.filter2D(R, -1, kernel=filter[:, :, 0])

    pic_new = cv2.merge([B_fil, G_fil, R_fil])
    # BGR -> RGB
    pic_new = pic_new[:,:,::-1]
    # resolving memory discontinuity problems
    pic_new = pic_new.copy()
    return pic_new

def my_fil_d4(pic):
    #b, c, h, w -> batch_size, channel, height, width
    b, c, h, w = pic.shape
    filter = np.array([[[0.25, 0., 0.25],
                        [0.5, 0.25, 0.5],
                        [0.25, 0., 0.25]],

                       [[0.5, 0.25, 0.5],
                        [1., 1., 1.],
                        [0.5, 0.25, 0.5]],

                       [[0.25, 0., 0.25],
                        [0.5, 0.25, 0.5],
                        [0.25, 0., 0.25]]])
    
    # pic -> numpy
    pic = pic.detach().cpu().numpy() if pic.requires_grad else pic.cpu().numpy()
    pic3 = pic
    for i in range(b):
        pic2 = pic[i]
        # CHW -> HWC, to split
        pic2 = pic2.transpose((1, 2, 0))
        # RGB -> BGR, to split
        pic2 = pic2[:,:,::-1]
        # resolving memory discontinuity problems
        pic2 = pic2.copy()
        
        [B, G, R] = cv2.split(pic2)
        
        B_fil = cv2.filter2D(B, -1, kernel=filter[:, :, 2])
        G_fil = cv2.filter2D(G, -1, kernel=filter[:, :, 1])
        R_fil = cv2.filter2D(R, -1, kernel=filter[:, :, 0])
        
        pic_new = cv2.merge([B_fil, G_fil, R_fil])
        # BGR -> RGB
        pic_new = pic_new[:,:,::-1]
        # resolving memory discontinuity problems
        pic_new = pic_new.copy()
        # HWC -> CHW
        pic_new = pic_new.transpose((2, 0, 1))
        pic3[i] = pic_new
    pic3 = torch.from_numpy(pic3)
    pic3 = pic3.cuda()
    return pic3
        
# processing the input images
class MyDataset(Dataset):
    def __init__(self, filepath, transform=None):
        self.filepath = filepath
        self.transform = transform
        
    def __getitem__(self, index):
        imgs = os.listdir(filepath)
        path = filepath + imgs[index]
        
        temp = cv2.imread(path)
        # BGR -> RGB
        temp = temp[:,:,::-1]
        # resolving memory discontinuity problems
        temp = temp.copy()
        temp = np.float64(temp)
        h, w, _ = temp.shape

        # crop the height and width of the input images to a multiple of 16
        # to avoid dimension mismatch problems in Up and Down Sampling
        temp = temp[0:(h - h%16), 0:(w-w%16), :]  

        # normalization
        temp = temp / 255.0
        
        label = temp
        data = my_fil(CFA(temp))
        
        if self.transform is not None:
            data = self.transform(data)
            label = self.transform(label)
            
        return data, label

    def __len__(self):
        return len(os.listdir(self.filepath))

# getting training data
def train_data_get(data_path):
    img_list = []
    labels = []
    imgs = os.listdir(data_path)

    for i in range(len(imgs)):
        path = data_path + imgs[i]
        temp = cv2.imread(path)
        # BGR -> RGB
        temp = temp[:,:,::-1]
        # resolving memory discontinuity problems
        temp = temp.copy()
        temp = np.float64(temp)
        h, w, _ = temp.shape

        # crop the height and width of the input images to a multiple of 16
        # to avoid dimension mismatch problems in Up and Down Sampling
        temp = temp[0:(h - h%16), 0:(w-w%16), :]  

        # normalization
        temp = temp / 255.0
        
        temp_label = temp
        temp = CFA(temp)
        temp = my_fil(temp)
        
        img_list.append(temp)
        labels.append(temp_label)

    train_list, train_label = img_list, labels
    print("Train data get complete.")
    return train_list, train_label

# getting validating data
def val_data_get(data_path):
    img_list = []
    labels = []
    imgs = os.listdir(data_path)

    for i in range(len(imgs)):
        path = data_path + imgs[i]
        temp = cv2.imread(path)
        # BGR -> RGB
        temp = temp[:,:,::-1]
        # resolving memory discontinuity problems
        temp = temp.copy()
        temp = np.float64(temp)
        h, w, _ = temp.shape

        # crop the height and width of the input images to a multiple of 16
        # to avoid dimension mismatch problems in Up and Down Sampling
        temp = temp[0:(h - h%16), 0:(w-w%16), :]
        
        # normalization
        temp = temp / 255.0
        
        temp_label = temp
        temp = CFA(temp)
        temp = my_fil(temp)
        
        img_list.append(temp)
        labels.append(temp_label)
        
    val_list, val_label = img_list, labels
    print("Validation data get complete.")
    return val_list, val_label

# processing datasets
transform = transforms.Compose([transforms.ToTensor()])
train_batch_size = 16
train_number_epoch = 50

train_dir1 = "/kaggle/input/div2k-dataset/DIV2K_train_HR/DIV2K_train_HR/"
val_dir = "/kaggle/input/div2k-dataset/DIV2K_valid_HR/DIV2K_valid_HR/"

trainset1 = MyDataset(train_dir1, transform=transform)
print("MyDataset Correct")
trainloader1 = DataLoader(trainset1, batch_size=train_batch_size, shuffle=True)
valset = MyDataset(val_dir, transform=transform)
valloader = DataLoader(valset, batch_size=1, shuffle=True)

# UNet network

# basic convolution model
class Conv(nn.Module):
    def __init__(self, C_in, C_out):
        super(Conv, self).__init__()
        self.layer = nn.Sequential(

            nn.Conv2d(C_in, C_out, 3, 1, 1),
            nn.LeakyReLU(),

            nn.Conv2d(C_out, C_out, 3, 1, 1),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.layer(x)


# down sampling model
class DownSampling(nn.Module):
    def __init__(self, C):
        super(DownSampling, self).__init__()
        self.Down = nn.Sequential(
            # 2x downsampling using convolution with the same number of channels
            nn.Conv2d(C, C, 3, 2, 1),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.Down(x)


# up sampling model
class UpSampling(nn.Module):

    def __init__(self, C):
        super(UpSampling, self).__init__()
        # feature map size expanded by 2x, number of channels halved
        self.Up = nn.Conv2d(C, C // 2, 1, 1)

    def forward(self, x, r):
        # Up-sampling using neighbor interpolation
        up = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.Up(up)
        # splicing, of the current upsampling, and of the previous downsampling process
        return torch.cat((x, r), 1)


# core network
class UNet(nn.Module):

    def __init__(self):
        super(UNet, self).__init__()

        self.C1 = Conv(3, 64)
        self.D1 = DownSampling(64)
        self.C2 = Conv(64, 128)
        self.D2 = DownSampling(128)
        self.C3 = Conv(128, 256)
        self.D3 = DownSampling(256)
        self.C4 = Conv(256, 512)
        self.D4 = DownSampling(512)
        self.C5 = Conv(512, 1024)

        # 4 up sampling models
        self.U1 = UpSampling(1024)
        self.C6 = Conv(1024, 512)
        self.U2 = UpSampling(512)
        self.C7 = Conv(512, 256)
        self.U3 = UpSampling(256)
        self.C8 = Conv(256, 128)
        self.U4 = UpSampling(128)
        self.C9 = Conv(128, 64)

        self.Th = torch.nn.Sigmoid()
        self.pred = torch.nn.Conv2d(64, 3, 3, 1, 1)

    def forward(self, x):
        # down sampling
        R1 = self.C1(x)
        R2 = self.C2(self.D1(R1))
        R3 = self.C3(self.D2(R2))
        R4 = self.C4(self.D3(R3))
        Y1 = self.C5(self.D4(R4))

        # up sampling
        # splicing while up sampling
        O1 = self.C6(self.U1(Y1, R4))
        O2 = self.C7(self.U2(O1, R3))
        O3 = self.C8(self.U3(O2, R2))
        O4 = self.C9(self.U4(O3, R1))
        return self.Th(self.pred(O4))
    
# initializing
net = UNet().cuda()
loss_func = nn.MSELoss()
best_train_loss = float('inf')
best_val_loss = float('inf')
lr = 1e-4
optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
train_dataloader = trainloader1
val_dataloader = valloader
epochs = train_number_epoch

# pre-training
def Pre_train(trainloader, valloader, model, loss_func, optimizer):
    global best_train_loss
    global best_val_loss
    device = torch.device('cuda')
    model = model.to(device)
    for epoch in range(0, epochs):
        # train part
        total_train_loss = 0
        counter1 = 0
        for i, data in enumerate(trainloader, 0):
            input_pic, label = data
            # transferred to GPU, type changed to float to match model parameter type
            input_pic, label = input_pic.to(device, dtype=torch.float), label.to(device, dtype=torch.float)
            
            
            # "output" has already been bilinearly interpolated
            # put into the network reconstruction to get a three-channel image.
            output = model(input_pic)

            optimizer.zero_grad()
            
            loss = loss_func(output, label)
            loss.requires_grad_(True)
            loss.backward()
            
            optimizer.step()
            total_train_loss += loss.item()
            counter1 += 1

        # val part
        model.eval()
        total_val_loss = 0
        
        # initializing three-channel PSNR value and CPSNR value
        total_psnr_r = 0
        total_psnr_g = 0
        total_psnr_b = 0
        total_cpsnr = 0
        
        counter2 = 0
        
        with torch.no_grad():
            for j, data2 in enumerate(valloader, 0):
                input_pic2, label2 = data2
                
                # CFA filtering and bilinear interpolation has been done in the "val_data_get" function
                input_pic2, label2 = input_pic2.to(device, dtype=torch.float), label2.to(device, dtype=torch.float)
                
                # put into the network
                output2 = model(input_pic2)
                
                # RGGB sampling
                output2 = output2.cuda()
                
                loss2 = loss_func(output2, label2)
                
                total_val_loss += loss2.item()
                
                # calculating MSE for the purpose of calculating PSNR
                
                # MSE values for RGB three channels
                
                mse_r = loss_func(output2[:, 0, :, :],
                                 label2[:, 0, :, :])
                mse_g = loss_func(output2[:, 1, :, :],
                                 label2[:, 1, :, :])
                mse_b = loss_func(output2[:, 2, :, :],
                                 label2[:, 2, :, :])
                
                # CPSNR, the average of the three-channel PSNR values
                cmse = loss_func(output2[:, :, :, :],
                                 label2[:, :, :, :])
                
                # calculating the sum of the PSNR values and CPSNR of the three RGB channels of all images
                if mse_r.item() < 1.0e-10:  # when MSR is too small
                    total_psnr_r += 100
                else:
                    total_psnr_r += 20 * math.log10(1 / math.sqrt(mse_r.item()))
                
                if mse_g.item() < 1.0e-10:  # when MSR is too small
                    total_psnr_g += 100
                else:
                    total_psnr_g += 20 * math.log10(1 / math.sqrt(mse_g.item()))
                
                if mse_b.item() < 1.0e-10:  # when MSR is too small
                    total_psnr_b += 100
                else:
                    total_psnr_b += 20 * math.log10(1 / math.sqrt(mse_b.item()))
                
                if cmse.item() < 1.0e-10:  # when MSR is too small
                    total_cpsnr += 100
                else:
                    total_cpsnr += 20 * math.log10(1 / math.sqrt(cmse.item()))
            
                counter2 += 1
        
                
            train_loss_avg = total_train_loss / counter1
            val_loss_avg = total_val_loss / counter2
            
            # calculating the average of the PSNR values and CPSNR of the three RGB channels of all images
            psnr_r_avg = total_psnr_r / counter2
            psnr_g_avg = total_psnr_g / counter2
            psnr_b_avg = total_psnr_b / counter2
            cpsnr_avg = total_cpsnr / counter2
            
            print("Epoch number: {} , Train loss: {:.4f}, Val loss: {:.4f}, \nPSNR_R: {:.4f}, PSNR_G: {:.4f}, PSNR_B: {:.4f}, CPSNR: {:.4f}\n".format(epoch, train_loss_avg, val_loss_avg, psnr_r_avg, psnr_g_avg, psnr_b_avg, cpsnr_avg))        

        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            torch.save(net.state_dict(), '/kaggle/working/model/best_pretrain_unet.pth')