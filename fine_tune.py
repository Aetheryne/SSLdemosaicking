# import libs
import numpy as np
import cv2
import os
import math

from PIL import Image

import argparse

import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader, Dataset

# parameters
parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', type=str, required=True)
parser.add_argument('--test_dir', type=str, required=True)
parser.add_argument('--checkpoint', type=str, required=True)
parser.add_argument('--batch_size', type=int, required=True)
parser.add_argument('--epoch_num', type=int, required=True)
parser.add_argument('--freeze', type=float, required=True)
parser.add_argument('--loss_pattern_num', type=int, default=3)
parser.add_argument('--save_results', type=bool, default=False)

args = parser.parse_args()

# create results folders
if args.save_results:
    os.makedirs('./results/original', exist_ok=True)
    os.makedirs('./results/mosaiced', exist_ok=True)
    os.makedirs('./results/fine_tuned', exist_ok=True)
    os.makedirs('./results/not_fine_tuned', exist_ok=True)

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

# translate RAW images to another pattern
def CFA_translate(pic: torch.Tensor, pattern:str):
    #b, c, h, w -> batch_size, channel, height, width
    b, c, h, w = pic.shape
    pic = pic.cuda()
    processed = torch.clone(pic)
    
    processed = processed.cuda()
    BGGR = np.array([[[0, 0, 1], [0, 1, 0]], [[0, 1, 0], [1, 0, 0]]])
    GBRG = np.array([[[0, 1, 0], [0, 0, 1]], [[1, 0, 0], [0, 1, 0]]])    
    GRBG = np.array([[[0, 1, 0], [1, 0, 0]], [[0, 0, 1], [0, 1, 0]]])
    time_h = int(np.ceil(h / 2))
    time_w = int(np.ceil(w / 2))
    if pattern == "BGGR":
        CFA = np.tile(BGGR, (time_h, time_w,1))
    elif pattern == "GBRG":
        CFA = np.tile(GBRG, (time_h, time_w,1))
    elif pattern == "GRBG":
        CFA = np.tile(GRBG, (time_h, time_w,1))
    CFA = CFA[:h,:w,:]
    #CFA -> h*w*3,RGB
    CFA3 = CFA.transpose((2,0,1))
    #CFA3 -> 3*h*w,RGB
    CFA3 = torch.from_numpy(CFA3)
    CFA3 = CFA3.cuda()
    
    for i in range(b):
        processed[i] = processed[i] * CFA3
        if pattern == "BGGR":
            processed[i] = torch.roll(processed[i],shifts=(1, 1), dims = (1,2))
        elif pattern == "GBRG":
            processed[i] = torch.roll(processed[i],shifts=(1, 0), dims = (1,2))
        elif pattern == "GBRG":
            processed[i] = torch.roll(processed[i],shifts=(0, 1), dims = (1,2))
    return processed

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

# processing the input Gehler-Shi images
class MyGehlerTrainDataset(Dataset):
    def __init__(self, filepath, transform=None):
        self.filepath = filepath
        self.transform = transform
        
    def __getitem__(self, index):
        imgs = os.listdir(self.filepath)
        path = self.filepath + '\\' + imgs[index]
        
        temp = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        # BGR -> RGB
        temp = temp[:,:,::-1]
        # resolving memory discontinuity problems
        temp = temp.copy()
        temp = np.float64(temp)
        h, w, _ = temp.shape
        
        if h == 2193 or h == 1460:
            temp = np.maximum(0., temp - 129.)
        
        temp = temp[:1024,:1024,:]

        # normalization
        temp = temp / 4095.0
        
        label = CFA(temp)
        data = my_fil(CFA(temp))
        
        if self.transform is not None:
            data = self.transform(data)
            label = self.transform(label)
            
        return data, label

    def __len__(self):
        return len(os.listdir(self.filepath))

# processing the input Gehler-Shi images
class MyGehlerTestDataset(Dataset):
    def __init__(self, filepath, transform=None):
        self.filepath = filepath
        self.transform = transform
        
    def __getitem__(self, index):
        imgs = os.listdir(self.filepath)
        path = self.filepath + '\\' + imgs[index]
        
        temp = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        # BGR -> RGB
        temp = temp[:,:,::-1]
        # resolving memory discontinuity problems
        temp = temp.copy()
        temp = np.float64(temp)
        h, w, _ = temp.shape
        
        if h == 2193 or h == 1460:
            temp = np.maximum(0., temp - 129.)
        
        temp = temp[:1024,:1024,:]

        # normalization
        temp = temp / 4095.0
        
        label = temp
        data = my_fil(CFA(temp))
        
        if self.transform is not None:
            data = self.transform(data)
            label = self.transform(label)
            
        return data, label

    def __len__(self):
        return len(os.listdir(self.filepath))

# processing datasets
transform = transforms.Compose([transforms.ToTensor()])
train_batch_size = args.batch_size
train_number_epoch = args.epoch_num

train_dir2 = args.train_dir
test_dir = args.test_dir

trainset2 = MyGehlerTrainDataset(train_dir2, transform=transform)
trainloader2 = DataLoader(trainset2, batch_size=train_batch_size, shuffle=False)
testset = MyGehlerTestDataset(test_dir, transform=transform)
testloader = DataLoader(testset, batch_size=1, shuffle=False)

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
net.load_state_dict(torch.load(args.checkpoint))
best_train_loss = float('inf')
best_test_loss = float('inf')
lr = 1e-5
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=1e-4)
train_dataloader = trainloader2
test_dataloader = testloader
epochs = train_number_epoch

# freeze paarameters
layers = len(list(net.named_parameters()))
for i, layer in enumerate(net.named_parameters(), 0):
    if i < args.freeze * layers:
        layer[1].requires_grad = False
print("{}% of the parameters are freezed".format(args.freeze * 100))

pat_num = args.loss_pattern_num

# fine tune
def SSL_train(trainloader, model, loss_func, optimizer):
    global best_train_loss
    device = torch.device('cuda')
    model = model.to(device)
    for epoch in range(0, epochs):
        # train part
        # model.train()
        
        total_train_loss = 0
        
        # initializing three-channel PSNR value and CPSNR value
        total_psnr_r = 0
        total_psnr_g = 0
        total_psnr_b = 0
        total_cpsnr = 0
        
        counter = 0
        for i, data in enumerate(trainloader, 0):
            input_pic, label = data
            # transferred to GPU, type changed to float to match model parameter type
            input_pic, label = input_pic.to(device, dtype=torch.float), label.to(device, dtype=torch.float)
            
            # "output" has already been bilinearly interpolated
            # put into the network reconstruction to get a three-channel image.
            output = model(input_pic)
            
            # sampling with pattern2 and bilinear interpolation
            output2 = CFA_translate(output, "BGGR")
                
            output2 = my_fil_d4(output2)
            
            output2 = model(output2)
            
            output2 = CFA_d4(output2, "BGGR")
            
            # sampling with pattern3 and bilinear interpolation
            output3 = CFA_translate(output, "GBRG")
            
            output3 = my_fil_d4(output3)
            
            output3 = model(output3)
            
            output3 = CFA_d4(output3, "GBRG")

            # sampling with pattern4 and bilinear interpolation
            output4 = CFA_translate(output, "GRBG")
            
            output4 = my_fil_d4(output4)
             
            output4 = model(output4)
            
            output4 = CFA_d4(output4, "GRBG")
            
            # setting the width of the boundary to be removed
            bound = 1
            
            label_b = label[:, :, bound:-bound,bound:-bound]
            output2_b = output2[:, :, bound:-bound,bound:-bound]
            output3_b = output3[:, :, bound:-bound,bound:-bound]
            output4_b = output4[:, :, bound:-bound,bound:-bound]
            
            optimizer.zero_grad()
            
            if pat_num == 1:
                loss = loss_func(output2_b, label_b)
            elif pat_num == 2:
                loss = loss_func(output2_b, label_b) + loss_func(output3_b, label_b)
            else:
                loss = loss_func(output2_b, label_b) + loss_func(output3_b, label_b) + loss_func(output4_b, label_b)
            loss.requires_grad_(True)
            loss.backward()
            
            optimizer.step()
            total_train_loss += loss.item()
            
            # calculating MSE for the purpose of calculating PSNR

            output = model(input_pic)
            
            output = CFA_d4(output, "RGGB")
            
            # MSE values for RGB three channels

            mse_r = loss_func(output[:, 0],
                             label[:, 0])
            mse_g = loss_func(output[:, 1],
                             label[:, 1])
            mse_b = loss_func(output[:, 2],
                             label[:, 2])
            
            # CPSNR, the average of the three-channel PSNR values
            cmse = loss_func(output,
                             label)

            # calculating the sum of the PSNR values and CPSNR of the three RGB channels of all images
            if mse_r.item() < 1.0e-10:  # when MSE is too small
                total_psnr_r += 100
            else:
                total_psnr_r += 20 * math.log10(1 / math.sqrt(mse_r.item()))

            if mse_g.item() < 1.0e-10:  # when MSE is too small
                total_psnr_g += 100
            else:
                total_psnr_g += 20 * math.log10(1 / math.sqrt(mse_g.item()))

            if mse_b.item() < 1.0e-10:  # when MSE is too small
                total_psnr_b += 100
            else:
                total_psnr_b += 20 * math.log10(1 / math.sqrt(mse_b.item()))

            if cmse.item() < 1.0e-10:  # when MSE is too small
                total_cpsnr += 100
            else:
                total_cpsnr += 20 * math.log10(1 / math.sqrt(cmse.item()))

            counter += 1

        train_loss_avg = total_train_loss / counter

        # calculating the average of the PSNR values and CPSNR of the three RGB channels of all images
        psnr_r_avg = total_psnr_r / counter
        psnr_g_avg = total_psnr_g / counter
        psnr_b_avg = total_psnr_b / counter
        cpsnr_avg = total_cpsnr / counter
        
        print("Epoch number: {} , Train loss: {:.4f}\nPSNR_R: {:.4f}, PSNR_G: {:.4f}, PSNR_B: {:.4f}, CPSNR: {:.4f}\n".format(epoch, train_loss_avg, psnr_r_avg, psnr_g_avg, psnr_b_avg, cpsnr_avg))
        if train_loss_avg < best_train_loss:
            best_train_loss = train_loss_avg
            torch.save(model.state_dict(), './checkpoints/fine_tuned.pth')

# test
def SSL_test(testloader, model, loss_func, save_result, SSLflag):
    device = torch.device('cuda')
    model = model.to(device)
    # test part
    model.eval()
    total_test_loss = 0

    # initializing three-channel PSNR value and CPSNR value
    total_psnr_r = 0
    total_psnr_g = 0
    total_psnr_b = 0
    total_cpsnr = 0

    counter = 0

    with torch.no_grad():
        for j, data in enumerate(testloader, 0):
            input_pic, label = data

            input_pic, label = input_pic.to(device, dtype=torch.float), label.to(device, dtype=torch.float)
            
            output = model(input_pic)
            if save_result:
                #1*3(RGB)*H*W -> H*W*3(RGB)
                labelImgTensor = label
                labelImg = labelImgTensor[0].detach().cpu().numpy().transpose((1,2,0))
                #rgb_range 1->255
                labelImg = np.uint8(labelImg*255.)
                labelImg = cv2.cvtColor(labelImg, cv2.COLOR_RGB2BGR)
                cv2.imwrite("./results/original/{}.png".format(j), labelImg)
                
                #1*3(RGB)*H*W -> H*W*3(RGB)
                labelImgTensor = CFA_d4(label, "RGGB")
                labelImg = labelImgTensor[0].detach().cpu().numpy().transpose((1,2,0))
                #rgb_range 1->255
                labelImg = np.uint8(labelImg*255.)
                labelImg = cv2.cvtColor(labelImg, cv2.COLOR_RGB2BGR)
                cv2.imwrite("./results/mosaiced/{}.png".format(j), labelImg)
            
                #1*3(RGB)*H*W -> H*W*3(RGB)
                img = output[0].detach().cpu().numpy().transpose((1,2,0))
                #rgb_range 1->255
                img = np.uint8(img*255.)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                if SSLflag:
                    cv2.imwrite("./results/fine_tuned/{}.png".format(j), img)
                else:
                    cv2.imwrite("./results/not_fine_tuned/{}.png".format(j), img)

            
            output = output.cuda()

            loss = loss_func(output, label)

            total_test_loss += loss.item()

            # calculating MSE for the purpose of calculating PSNR
                
            # MSE values for RGB three channels

            mse_r = loss_func(output[:, 0, :, :],
                             label[:, 0, :, :])
            mse_g = loss_func(output[:, 1, :, :],
                             label[:, 1, :, :])
            mse_b = loss_func(output[:, 2, :, :],
                             label[:, 2, :, :])
            
            # CPSNR, the average of the three-channel PSNR values
            cmse = loss_func(output, label)

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

            counter += 1

        test_loss_avg = total_test_loss / counter

        # calculating the average of the PSNR values and CPSNR of the three RGB channels of all images
        psnr_r_avg = total_psnr_r / counter
        psnr_g_avg = total_psnr_g / counter
        psnr_b_avg = total_psnr_b / counter
        cpsnr_avg = total_cpsnr / counter

        print("Test loss: {:.4f}\nPSNR_R: {:.4f}, PSNR_G: {:.4f}, PSNR_B: {:.4f}, CPSNR: {:.4f}\n".format(test_loss_avg, psnr_r_avg, psnr_g_avg, psnr_b_avg, cpsnr_avg))        

print("The Test loss and PSNR before SSL:")

if args.save_results:
    SSL_test(test_dataloader, net, loss_func, True, False)
else:
    SSL_test(test_dataloader, net, loss_func, False, False)

SSL_train(train_dataloader, net, loss_func, optimizer)

net.load_state_dict(torch.load('./checkpoints/fine_tuned.pth'))

print("The Test loss and PSNR after SSL:")

if args.save_results:
    SSL_test(test_dataloader, net, loss_func, True, True)
else:
    SSL_test(test_dataloader, net, loss_func, False, True)