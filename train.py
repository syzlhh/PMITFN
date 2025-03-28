from __future__ import print_function
import argparse
import os
from math import log10
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from math import exp
from dataset import CityDataset2
from Encoder_new import Encoder_NewBlock_fusion_with_T625
from torch.optim.lr_scheduler import MultiStepLR
from torch.autograd import Variable
from skimage.measure.simple_metrics import compare_psnr

use_cuda = torch.cuda.is_available()
import cv2


def batch_PSNR(img, imclean, data_range=None):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i,:,:,:], Img[i,:,:,:])
    return (PSNR/Img.shape[0])

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

# from utils import *

parser = argparse.ArgumentParser(description="Pytorch dehaze")
parser.add_argument("--lr", type=float, default=0.1, help="Learning Rate, Default=0.1")
parser.add_argument("--step", type=int, default=5, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default=5")
parser.add_argument("--clip", type=float, default=0.01, help="Clipping Gradients, Default=0.01")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default=0.9")
parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float, help="Weight decay, Default=1e-4")
opt = parser.parse_args()
save_path = os.path.join("checkpoint", 'city_model')


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def adjust_learning_rate(optimizer, epoch,decay=0.75):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = opt.lr * (decay ** (epoch // opt.step))
    return lr


def train():
    torch.manual_seed(1334)
    torch.cuda.manual_seed(1334)
    np.random.seed(1334)

    train_data = CityDataset2('D:\Data\Data\leftImg8bit_trainvaltest_foggy\leftImg8bit_foggy\\',
                             'D:\Data\Data\leftImg8bit_trainvaltest\leftImg8bit\\',
                              'D:\Data\Data\leftImg8bit_trainvaltest_transmittance\leftImg8bit_transmittance\\')
    test_data = CityDataset2('D:\Data\Data\leftImg8bit_trainvaltest_foggy\leftImg8bit_foggy\\',
                            'D:\Data\Data\leftImg8bit_trainvaltest\leftImg8bit\\', istrain=False)
    training_data_loader = DataLoader(dataset=train_data, num_workers=4, batch_size=1,
                                      shuffle=True)
    testing_data_loader = DataLoader(dataset=test_data, num_workers=4, batch_size=1,
                                     shuffle=False)
    device = torch.device("cuda:0" if use_cuda else "cpu")

    net = Encoder_NewBlock_fusion_with_T625().to(device)
    criterionL1 = nn.L1Loss(size_average=True).to(device)
    criterionMSE = nn.MSELoss().to(device)

    criterion = SSIM()
    optimizer = optim.Adam(net.parameters(), lr=0.0002)
    scheduler = MultiStepLR(optimizer, milestones=[5, 10,15,20,25], gamma=0.75)
    # optimizer = optim.SGD(net.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
    old_psnr = 0
    for epoch in range(30):
        for iteration, batch in enumerate(training_data_loader, 1):
            real_a, real_b,real_t = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            real_b_8x = F.interpolate(real_b,scale_factor=1/8,mode='bilinear')
            real_b_4x = F.interpolate(real_b, scale_factor=1/4, mode='bilinear')
            real_b_2x = F.interpolate(real_b, scale_factor=1/2, mode='bilinear')
            real_t_8x = F.interpolate(real_t, scale_factor=1 / 8, mode='bilinear')
            real_t_4x = F.interpolate(real_t, scale_factor=1 / 4, mode='bilinear')
            real_t_2x = F.interpolate(real_t, scale_factor=1 / 2, mode='bilinear')

            fake_b,fake_t,fake_b8,fake_t8,fake_b4,fake_t4,fake_b2,fake_t2 = net(real_a)

            optimizer.zero_grad()
            loss_d = -criterion(fake_b / 2. + 0.5, real_b / 2. + 0.5)

            loss_d8 = -criterion(fake_b8 / 2. + 0.5, real_b_8x / 2. + 0.5)
            loss_d4 = -criterion(fake_b4 / 2. + 0.5, real_b_4x / 2. + 0.5)
            loss_d2 = -criterion(fake_b2 / 2. + 0.5, real_b_2x / 2. + 0.5)
            loss_t = criterionL1(fake_t , real_t )
            loss_t8 = criterionL1(fake_t8 , real_t_8x )
            loss_t4 = criterionL1(fake_t4 , real_t_4x )
            loss_t2 = criterionL1(fake_t2 , real_t_2x )

            loss = (loss_d + 0.2 * loss_d8 + 0.3 * loss_d4 + 0.5 * loss_d2)+\
                   (loss_t + 0.2 * loss_t8 + 0.3 * loss_t4 + 0.5 * loss_t2)

            loss.backward()

            optimizer.step()
            scheduler.step(epoch)
            if iteration % 100 == 0:
                print("===> Epoch[{}]({}/{}): Loss_d: {:.4f} Loss_d8: {:.4f} Loss_d4: {:.4f} Loss_d2: {:.4f}".format(
                    epoch, iteration, len(training_data_loader), loss_d.item(), loss_d8.item(), loss_d4.item(),loss_d2.item()))
                print("===> Epoch[{}]({}/{}): Loss_T: {:.4f} Loss_T8: {:.4f} Loss_T4: {:.4f} Loss_T2: {:.4f}".format(
                    epoch, iteration, len(training_data_loader), loss_t.item(), loss_t8.item(), loss_t4.item(),
                    loss_t2.item()))
            if iteration % 500 == 0:
                out_hazy = (real_a[0]).cpu().detach().permute(1, 2, 0).numpy()
                out_gt = (real_b[0]).cpu().detach().permute(1, 2, 0).numpy()
                out_weight = (real_t[0]).cpu().detach().permute(1, 2, 0).numpy()
                fake_out = (fake_b[0]).cpu().detach().permute(1, 2, 0).numpy()
                fake_t = (fake_t[0]).cpu().detach().permute(1, 2, 0).numpy()

                fake_out = np.clip(fake_out,-1,1)
                fake_t = np.clip(fake_t, -1, 1)

                out_hazy = ((out_hazy + 1) / 2 * 255).astype(np.uint8)
                out_gt = ((out_gt + 1) / 2 * 255).astype(np.uint8)
                out_weight = ((out_weight+1)/2 * 255).astype(np.uint8)
                fake_out = ((fake_out + 1) / 2 * 255).astype(np.uint8)
                fake_t = ((fake_t + 1) / 2 * 255).astype(np.uint8)

                out_hazy = cv2.cvtColor(out_hazy, cv2.COLOR_RGB2BGR)
                out_gt = cv2.cvtColor(out_gt, cv2.COLOR_RGB2BGR)
                fake_out = cv2.cvtColor(fake_out, cv2.COLOR_RGB2BGR)

                cv2.imwrite("D:\\pytorch_model\\torch_dehaze\\samples\\{}_hazy.png".format(iteration), out_hazy)
                cv2.imwrite("D:\\pytorch_model\\torch_dehaze\\samples\\{}_gt.png".format(iteration), out_gt)
                cv2.imwrite("D:\\pytorch_model\\torch_dehaze\\samples\\{}_weight.png".format(iteration), out_weight)
                cv2.imwrite("D:\\pytorch_model\\torch_dehaze\\samples\\{}_out.png".format(iteration), fake_out)
                cv2.imwrite("D:\\pytorch_model\\torch_dehaze\\samples\\{}_T.png".format(iteration), fake_t)

            if iteration % 4000 == 0:
                with torch.no_grad():
                    net.eval()
                    avg_psnr = 0
                    for batch in testing_data_loader:
                        input, target = batch[0].to(device), batch[1].to(device)
                        prediction = net(input)
                        psnr = batch_PSNR(prediction[0], target)
                        avg_psnr += psnr
                    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))
                    if avg_psnr>old_psnr:
                        if not os.path.exists("checkpoint"):
                            os.mkdir("checkpoint")
                        if not os.path.exists(os.path.join("checkpoint", 'city_model')):
                            os.mkdir(os.path.join("checkpoint", 'city_model'))
                        model_out_path = "checkpoint/{}/netG_model_{}_{}.pth".format('city_model',epoch,avg_psnr / len(testing_data_loader))
                        torch.save(net, model_out_path)
                        old_psnr = avg_psnr
                        print("Checkpoint saved to {}".format("checkpoint" + 'in_outdoor'))
        if epoch % 1 == 0:
            if not os.path.exists("checkpoint"):
                os.mkdir("checkpoint")
            if not os.path.exists(os.path.join("checkpoint", 'city_model')):
                os.mkdir(os.path.join("checkpoint", 'city_model'))

            with torch.no_grad():
                net.eval()
                avg_psnr = 0
                for batch in testing_data_loader:
                    input, target = batch[0].to(device), batch[1].to(device)
                    prediction = net(input)
                    psnr = batch_PSNR(prediction[0], target)
                    avg_psnr += psnr
                print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))
            model_out_path = "checkpoint/{}/netG_model_{}_{}.pth".format('city_model', epoch,
                                                                         avg_psnr / len(testing_data_loader))
            torch.save(net, model_out_path)
            print("Checkpoint saved to {}".format("checkpoint" + 'city_model'))


if __name__ == '__main__':
    train()