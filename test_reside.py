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

from torchvision import utils as uvtils
import torchvision.transforms as transforms
use_cuda = torch.cuda.is_available()
from torch.autograd import Variable
from glob import glob
import time
import cv2


logdir = '..\\model\\'
which_model= 'netG_model_its.pth'

save_path_out='./outdoor/'
save_path_in='./indoor/'

def normalize(data):
    return data / 255.
def test():
    device = torch.device("cuda:0" if use_cuda else "cpu")

    net = torch.load(logdir+which_model)
    if use_cuda:
        net = net.cuda()
   
    net.eval()
   

    time_test = 0
    count = 0
    imgs_hazy = []
    imgs_clean = []
    imgs_hazy = glob('D:\SOTS\outdoor\hazy\\*.jpg')
    test_gt_dir = imgs_hazy
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
           
            
            fake_b = net(hazy_img)+hazy_img
          
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
           
            cv2.imwrite(save_path_out+img_name, save_out)

            count += 1
    print('Avg. time:', time_test / count)
	
def test_its():
    device = torch.device("cuda:0" if use_cuda else "cpu")

    net = torch.load(logdir+which_model)
    if use_cuda:
        net = net.cuda()
  
    net.eval()
   
    time_test = 0
    count = 0
    imgs_hazy = []
    imgs_clean = []
    imgs_hazy = glob('D:\SOTS\indoor\\nyuhaze500\hazy\\*.png')
    test_gt_dir = imgs_hazy
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
        hazy_img = Variable(torch.Tensor(hazy_img))
        if use_cuda:
            hazy_img = hazy_img.cuda()
          

        with torch.no_grad():
            if use_cuda:
                torch.cuda.synchronize()
            start_time = time.time()
            
            fake_b = net(hazy_img)

            if use_cuda:
                torch.cuda.synchronize()
            end_time = time.time()
            dur_time = end_time - start_time
            time_test += dur_time
            if use_cuda:
                save_out = np.uint8(127.5 *( fake_b.data.cpu().numpy().squeeze()+1))
            else:
                save_out = np.uint8(255.0 * (fake_b).data.numpy().squeeze())
            save_out = save_out.transpose(1, 2, 0)

            b, g, r = cv2.split(save_out)
            save_out = cv2.merge([r, g, b])
            save_out = cv2.resize(save_out,(w,h))
            cv2.imwrite(save_path_in+img_name, save_out)

            count += 1
    print('Avg. time:', time_test / count)



if __name__ =='__main__':
    # test()
    test_its()
   