from __future__ import print_function
import argparse
import os
from math import log10
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
# from networks import Generator
from torch.autograd import Variable
import scipy.misc as misc
from torchvision import utils as uvtils
use_cuda = torch.cuda.is_available()

from glob import glob
import time
import cv2

#the dir of the model checkpoint
logdir = 'D:\\pytorch_model\\torch_dehaze\\upload_PR\\model\\'
which_model= 'netG_model_foggy_city.pth'

test_hazy_dir = 'D:\Data\Data\leftImg8bit_trainvaltest_foggy\leftImg8bit_foggy\\'
test_gt_dir = 'D:\Data\Data\leftImg8bit_trainvaltest_foggy\leftImg8bit_foggy\\'
test_list= '\\test_list.txt'
save_path = './valid_foggy_city/'
save_path_real = './sample_out/real_city/'

def normalize(data):
    return data / 255.
def test_city():
    device = torch.device("cuda:0" if use_cuda else "cpu")

    net = torch.load(logdir+which_model)
    if use_cuda:
        net = net.cuda()
    # net.to(device)
    net.eval()
    # if True:
    #     checkpoint = "checkpoint/{}/netG_model_epoch_{}.pth".format('in_outdoor', epoch)
    #     checkpoint = torch.load(args.resume)

    time_test = 0
    count = 0
    imgs_hazy = []
    imgs_clean = []
    with open('D:\Data\Data\leftImg8bit_trainvaltest_foggy\leftImg8bit_foggy\\val_list.txt', 'r') as f:
        lines = f.readlines()
        for img in lines:
            img = img.strip('\n')
            hazy = img.split(' ')[0]
            gt = img.split(' ')[-1]
            imgs_hazy.append(test_hazy_dir + hazy)
            # name = clean_data_dir+img.split("_")[0]+'.jpg'
            imgs_clean.append(test_gt_dir + gt)
    data = zip(imgs_hazy,imgs_clean)
    for hazy_name,gt_name in data:
        hazy_img = cv2.imread(hazy_name,cv2.IMREAD_GRAYSCALE)
        t = hazy_img.shape
        h = t[0]
        w = t[1]
        hazy_img = cv2.resize(hazy_img,(768,512))
        
        img_name = hazy_name.split('\\')[-1]

        b, g, r = cv2.split(hazy_img)
        hazy_img = cv2.merge([r, g, b])
       
        hazy_img = normalize(np.float32(hazy_img))*2-1
       
        hazy_img = np.expand_dims(hazy_img.transpose(2, 0, 1), 0)
        hazy_img = np.expand_dims(hazy_img,0)
        hazy_img = Variable(torch.Tensor(hazy_img))
        if use_cuda:
            hazy_img = hazy_img.cuda()
            # hazy_img.to(device)

        with torch.no_grad():
            if use_cuda:
                torch.cuda.synchronize()
            start_time = time.time()
           
            fake_b = net(hazy_img)
            
            fake_b = torch.clamp(fake_b[0], -1., 1.)
            if use_cuda:
                torch.cuda.synchronize()
            end_time = time.time()
            dur_time = end_time - start_time
            time_test += dur_time
            if use_cuda:
                save_out = np.uint8(127.5 *( fake_b.data.cpu().numpy().squeeze()+1))
            else:
                save_out = np.uint8(127.5 * (fake_b+1).data.numpy().squeeze())
            save_out = save_out.transpose(1, 2, 0)
           
            b, g, r = cv2.split(save_out)
            save_out = cv2.merge([r, g, b])
           
            save_out = cv2.resize(save_out,(2048,1024))
            cv2.imwrite(save_path+img_name, save_out)

            count += 1
    print('Avg. time:', time_test / count)
def test_real():
    device = torch.device("cuda:0" if use_cuda else "cpu")

    net = torch.load(logdir+which_model)
    if use_cuda:
        net = net.cuda()
   
    net.eval()
   

    time_test = 0
    count = 0
    imgs_hazy = []
    imgs_clean = []
    imgs_hazy = glob('D:\\Data\\inputs\\*.png')
    for img in imgs_hazy:

        name = img.split('\\')[-1].split('_')[0]
        gt = name+'.png'
        imgs_clean.append(test_gt_dir + gt)
    data = zip(imgs_hazy,imgs_clean)
    for hazy_name,gt_name in data:
        hazy_img = cv2.imread(hazy_name)
        t = hazy_img.shape
        h = t[0]
        w = t[1]
        hazy_img = cv2.resize(hazy_img,((w//8)*8,(h//8)*8))
        
        img_name = hazy_name.split('\\')[-1]
        b, g, r = cv2.split(hazy_img)
        hazy_img = cv2.merge([r, g, b])
       
        hazy_img = normalize(np.float32(hazy_img))*2-1
        hazy_img = np.expand_dims(hazy_img.transpose(2, 0, 1), 0)
        hazy_img = torch.Tensor(hazy_img)
        if use_cuda:
            hazy_img = hazy_img.cuda()
            # hazy_img.to(device)

        with torch.no_grad():
            if use_cuda:
                torch.cuda.synchronize()
            start_time = time.time()
           
            
            fake_b = net(hazy_img)
          
            fake_b = torch.clamp(fake_b[0], -1, 1.)
           
            if use_cuda:
                torch.cuda.synchronize()
            end_time = time.time()
            dur_time = end_time - start_time
            time_test += dur_time
            if use_cuda:
                save_out = np.uint8(127.5 * (fake_b.data.cpu().numpy().squeeze() + 1))
             
            else:
                save_out = np.uint8(255.0 * (fake_b).data.numpy().squeeze())
            save_out = save_out.transpose(1, 2, 0)

            b, g, r = cv2.split(save_out)
            save_out = cv2.merge([r, g, b])
           
            cv2.imwrite(save_path_real+img_name, save_out)

            count += 1
    print('Avg. time:', time_test / count)

if __name__ =='__main__':
    #test_city()
    test_real()
   