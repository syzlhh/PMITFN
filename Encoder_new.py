import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import functools
from Blocks import *
from cbam import *


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,dilation=1):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride,dilation=dilation)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out

class UpsampleConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
      super(UpsampleConvLayer, self).__init__()
      reflection_padding = kernel_size // 2
      self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
      self.conv2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out





class Encoder_NewBlock_fusion_with_T625(nn.Module):
    def __init__(self, channel=64, conv=default_conv, res_blokcs=8):
        super(Encoder_NewBlock_fusion_with_T625, self).__init__()
        self.channels = channel
        self.conv_d1 = nn.Conv2d(3, self.channels, 7, 1, 3)
        # self.res_blk_d1 = nn.Sequential(
        #     Res_in_res_blk(self.channels, self.channels, dialation=1),
        #     Res_in_res_blk(self.channels, self.channels, dialation=1),
        # )
        # self.fa_d1 = SpatialFusionAttentionLayer(self.channels)
        self.conv_d2 = nn.Conv2d(self.channels, self.channels * 2, 4, 2, 1)
        # self.res_blk_d2 = nn.Sequential(
        #     Res_in_res_blk(self.channels, self.channels, dialation=1),
        #     Res_in_res_blk(self.channels, self.channels, dialation=1),
        # )
        # self.fa_d2 = SpatialFusionAttentionLayer(self.channels)
        self.conv_d3 = nn.Conv2d(self.channels * 2, self.channels * 4, 4, 2, 1)
        # self.res_blk_d3 = nn.Sequential(
        #     Res_in_res_blk(self.channels*2, self.channels*2, dialation=1),
        #     Res_in_res_blk(self.channels*2, self.channels*2, dialation=1),
        # )
        # self.fa_d3 = SpatialFusionAttentionLayer(self.channels*2)
        self.conv_d4 = nn.Conv2d(self.channels * 4, self.channels * 8, 4, 2, 1)
        # self.conv_d5 = nn.Conv2d(self.channels * 4, self.channels * 8, 4, 2, 1)
        # self.act = nn.ReLU(inplace=True)
        self.res_blk0 = ResBlock(default_conv, self.channels * 8, 3, act=nn.ReLU())
        self.res_blk1 = ResBlock(default_conv, self.channels * 8, 3, act=nn.ReLU())
        self.res_blk2 = ResBlock(default_conv, self.channels * 8, 3, act=nn.ReLU())

        self.res_blk3 = ResBlock(default_conv, self.channels * 8, 3, act=nn.ReLU())
        self.res_blk4 = ResBlock(default_conv, self.channels * 8, 3, act=nn.ReLU())
        self.res_blk5 = ResBlock(default_conv, self.channels * 8, 3, act=nn.ReLU())

        self.res_blk6 = ResBlock(default_conv, self.channels * 8, 3, act=nn.ReLU())
        self.res_blk7 = ResBlock(default_conv, self.channels * 8, 3, act=nn.ReLU())
        self.res_blk8 = ResBlock(default_conv, self.channels * 8, 3, act=nn.ReLU())

        self.res_blk9 = ResBlock(default_conv, self.channels * 4, 3, act=nn.ReLU())
        self.res_blk10 = ResBlock(default_conv, self.channels * 4, 3, act=nn.ReLU())

        self.res_blk11 = ResBlock(default_conv, self.channels * 4, 3, act=nn.ReLU())
        self.res_blk12 = ResBlock(default_conv, self.channels * 4, 3, act=nn.ReLU())

        self.res_blk13 = ResBlock(default_conv, self.channels * 2, 3, act=nn.ReLU())
        self.res_blk14 = ResBlock(default_conv, self.channels * 2, 3, act=nn.ReLU())

        self.res_blk15 = ResBlock(default_conv, self.channels, 3, act=nn.ReLU())

        self.relu = nn.ReLU()
        self.conv1X11 = nn.Conv2d(self.channels * 32, self.channels * 8, 1, 1, 0)

        self.pa8x_s_m = ChannelAttention(self.channels*8)
        self.pa8x_s_t = ChannelAttention(self.channels * 8)
        self.pa8x_c_m = nn.Sequential(
            nn.Conv2d(self.channels * 8, self.channels * 8, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.channels * 8, self.channels * 8, 3, 1, 1),
            nn.Sigmoid()
        )
        self.pa8x_c_t = nn.Sequential(
            nn.Conv2d(self.channels * 8, self.channels * 8, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.channels * 8, self.channels * 8, 3, 1, 1),
            nn.Sigmoid()
        )
        # self.conv8x_11 = nn.Conv2d(self.channels * 16, self.channels * 8, 1, 1, 0)
        # self.fa5 = FusionAttentionLayer(self.channels * 8)
        self.conv8x_I = ResBlock(default_conv, self.channels * 8, 3, act=nn.ReLU())
        self.conv8x_T = ResBlock(default_conv, self.channels * 8, 3, act=nn.ReLU())

        self.convd8x_out = nn.Conv2d(self.channels * 8, 3, 3, 1, 1)
        self.convd8x_out_T = nn.Conv2d(self.channels * 8, 1, 3, 1, 1)
        self.convd8x_DI = nn.Sequential(
            nn.Conv2d(self.channels * 12, self.channels * 4, 1, 1, 0),
            nn.Conv2d(self.channels * 4, self.channels * 4, 3, 1, 1),
        )
        self.convd8x_DT = nn.Sequential(
            nn.Conv2d(self.channels * 12, self.channels * 4, 1, 1, 0),
            nn.Conv2d(self.channels * 4, self.channels * 4, 3, 1, 1),
        )

        self.conv4x_I = ResBlock(default_conv, self.channels * 4, 3, act=nn.ReLU())
        self.conv4x_T = ResBlock(default_conv, self.channels * 4, 3, act=nn.ReLU())
        self.convd4x_out = nn.Conv2d(self.channels * 4, 3, 3, 1, 1)
        self.convd4x_out_T = nn.Conv2d(self.channels * 4, 1, 3, 1, 1)
        # self.convd8x_out = nn.Conv2d(self.channels * 4, 3, 3, 1, 1)
        # self.convd8x_out_T = nn.Conv2d(self.channels * 4, 1, 3, 1, 1)
        self.convd4x_DI = nn.Sequential(
            nn.Conv2d(self.channels * 6, self.channels * 2, 1, 1, 0),
            nn.Conv2d(self.channels * 2, self.channels * 2, 3, 1, 1),
        )
        self.convd4x_DT = nn.Sequential(
            nn.Conv2d(self.channels * 6, self.channels * 2, 1, 1, 0),
            nn.Conv2d(self.channels * 2, self.channels * 2, 3, 1, 1),
        )

        self.pa4x_s_m = ChannelAttention(self.channels * 4)
        self.pa4x_s_t = ChannelAttention(self.channels * 4)
        self.pa4x_c_m = nn.Sequential(
            nn.Conv2d(self.channels * 4, self.channels * 4, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.channels * 4, self.channels * 4, 3, 1, 1),
            nn.Sigmoid()
        )
        self.pa4x_c_t = nn.Sequential(
            nn.Conv2d(self.channels * 4, self.channels * 4, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.channels * 4, self.channels * 4, 3, 1, 1),
            nn.Sigmoid()
        )
        # self.conv4x_11 = nn.Conv2d(self.channels * 8, self.channels * 4, 1, 1, 0)
        # self.convd4x_out = nn.Conv2d(self.channels*5// 4, 3, 3, 1, 1)
        # self.convd4x_out_T = nn.Conv2d(self.channels*5// 4, 1, 3, 1, 1)

        self.conv2x_I = ResBlock(default_conv, self.channels * 2, 3, act=nn.ReLU())
        self.conv2x_T = ResBlock(default_conv, self.channels * 2, 3, act=nn.ReLU())
        self.convd2x_out = nn.Conv2d(self.channels * 2, 3, 3, 1, 1)
        self.convd2x_out_T = nn.Conv2d(self.channels * 2, 1, 3, 1, 1)
        self.convd2x_DI = nn.Sequential(
            nn.Conv2d(self.channels * 3, self.channels, 1, 1, 0),
            nn.Conv2d(self.channels, self.channels, 3, 1, 1),
        )
        self.convd2x_DT = nn.Sequential(
            nn.Conv2d(self.channels * 3, self.channels, 1, 1, 0),
            nn.Conv2d(self.channels, self.channels, 3, 1, 1),
        )

        self.pa2x_s_m = ChannelAttention(self.channels*2 )
        self.pa2x_s_t = ChannelAttention(self.channels*2 )
        self.pa2x_c_m =nn.Sequential(
            nn.Conv2d(self.channels * 2, self.channels * 2, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.channels * 2, self.channels * 2, 3, 1, 1),
            nn.Sigmoid()
        )
        self.pa2x_c_t = nn.Sequential(
            nn.Conv2d(self.channels * 2, self.channels * 2, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.channels * 2, self.channels * 2, 3, 1, 1),
            nn.Sigmoid()
        )
        # self.conv2x_11 = nn.Conv2d(self.channels * 4, self.channels*2, 1, 1,0)
        self.convx_I = ResBlock(default_conv, self.channels, 3, act=nn.ReLU())
        self.convx_T = ResBlock(default_conv, self.channels, 3, act=nn.ReLU())
        # self.pax = nn.Sequential(
        #     nn.Conv2d(self.channels * 2, self.channels, 1, 1, 0),
        #     nn.ReLU(),
        #     nn.Conv2d(self.channels, 1, 3, 1, 1),
        #     nn.Sigmoid()
        # )
        self.pax_s_m = ChannelAttention(self.channels )
        self.pax_s_t = ChannelAttention(self.channels )
        self.pax_c_m = SpatialAttention()
        self.pax_c_t = SpatialAttention()
        # self.convd2x_out = nn.Conv2d(self.channels *5// 8, 3, 3, 1, 1)
        # self.convd2x_out_T = nn.Conv2d(self.channels*5 // 8, 1, 3, 1, 1)
        self.conv_last = ConvLayer(self.channels, 3, kernel_size=7, stride=1)
        self.convd_T = ConvLayer(self.channels, 1, kernel_size=7, stride=1)

        self.weight1 = nn.Parameter(torch.Tensor([0]))
        self.weight2 = nn.Parameter(torch.Tensor([0]))
        self.weight3 = nn.Parameter(torch.Tensor([0]))
        self.weight4 = nn.Parameter(torch.Tensor([0]))

    def forward(self, in_x):
        encoder_layer = []
        x = self.conv_d1(in_x)
        # x_d1 = self.res_blk_d1(x)
        # x = self.fa_d1(x_d1, x)
        encoder_layer.append(x)
        x = self.conv_d2(self.relu(x))
        # x_d2 = self.res_blk_d2(x)
        # x = self.fa_d2(x_d2,x)
        encoder_layer.append(x)
        x = self.conv_d3(self.relu(x))
        # x_d3 = self.res_blk_d3(x)
        # x = self.fa_d3(x_d3, x)
        encoder_layer.append(x)
        x = self.conv_d4(self.relu(x))
        encoder_layer.append(x)
        N, C, H, W = x.size()
        x0 = self.res_blk0(x)
        x1 = self.res_blk1(x0)
        x2 = self.res_blk2(x1)
        x3 = self.res_blk3(x2)
        x4 = self.res_blk4(x3)
        x5 = self.res_blk5(x4)
        x6 = self.res_blk6(x5)
        x7 = self.res_blk7(x6)
        x8 = self.res_blk8(x7)

        res8x = self.conv1X11(self.relu(torch.cat([x, x8, x2, x5], dim=1)))
        res8x_I = self.conv8x_I(self.relu(res8x))

        res8x_T = self.conv8x_T(self.relu(res8x))

        # res8x_Ta = res8x_T + res8x_I

        res8x_I1 = self.pa8x_c_t(res8x_T) * res8x_I
        res8x_FI = self.pa8x_s_t(res8x_I1) * res8x_I1
        res8x_FI = res8x_FI+res8x_I

        res8x_I2 = self.pa8x_c_m(res8x_I) * res8x_T
        res8x_FT = self.pa8x_s_m(res8x_I2) * res8x_I2
        res8x_FT = res8x_FT + res8x_T

        # res8x_I = self.conv8x_11(torch.cat((res8x_I2,res8x_I1),dim=1))
        res8x_out_T = self.convd8x_out_T(self.relu(res8x_FT))

        # real_b_8x = F.interpolate(in_x, scale_factor=1 / 8, mode='bilinear')

        # res8x_out_M = 2*(real_b_8x+1)/(res8x_out_T+1+(10**-10))-(res8x_out_T+1)-1
        # res8x_out_M = (real_b_8x - 1) / (res8x_out_T + 1 + (10 ** -10)) + 1
        # res8x_out_M = res8x_out_M * 2 - 1
        # res8x_out_I = self.convd8x_out(self.relu(res8x_FI))
        res8x_out = self.convd8x_out(self.relu(res8x_FI))


        res8x_FI = F.interpolate(res8x_FI, encoder_layer[-2].size()[2:], mode='bilinear')
        res8x_Up = F.interpolate(res8x_out, encoder_layer[-2].size()[2:], mode='bilinear')
        res8x_FT = F.interpolate(res8x_FT, encoder_layer[-2].size()[2:], mode='bilinear')
        cl_4 = self.res_blk9(encoder_layer[-2])
        cl_41 = self.res_blk10(cl_4)
        cl_43 = self.res_blk11(cl_41)
        cl_44 = self.res_blk12(cl_43)
        res4x_I = self.conv4x_I(self.relu(cl_44))
        res4x_T = self.conv4x_T(self.relu(cl_44))

        res4x_I = torch.cat([res8x_FI, res4x_I], dim=1)
        res4x_I = self.convd8x_DI(self.relu(res4x_I))

        res4x_T = torch.cat([res8x_FT, res4x_T], dim=1)
        res4x_T = self.convd8x_DT(self.relu(res4x_T))

        # res4x_Ta = res4x_I + res4x_T
        res4x_I1 = self.pa4x_c_t(res4x_T) * res4x_I
        res4x_FI = self.pa4x_s_t(res4x_I1) * res4x_I1+res4x_I

        res4x_I2 = self.pa4x_c_m(res4x_I) * res4x_T
        res4x_FT = self.pa4x_s_m(res4x_I2) * res4x_I2+res4x_T
        # res4x_I = self.conv4x_11(torch.cat([res4x_I2, res4x_I1], dim=1))

        res4x_out_T =  self.convd4x_out_T(self.relu(res4x_FT))

        # real_b_4x = F.interpolate(in_x, scale_factor=1 / 4, mode='bilinear')
        # res4x_out_M = 2 * (real_b_4x + 1) / (res4x_out_T + 1 + (10 ** -10)) - (res4x_out_T + 1) - 1
        # res4x_out_M = (real_b_4x - 1) / (res4x_out_T + 1 + (10 ** -10)) + 1
        # res4x_out_M = res4x_out_M * 2 - 1
        res4x_out_I = self.convd4x_out(self.relu(res4x_FI))
        res4x_out = (1 - self.weight2) * res4x_out_I + self.weight2 * (res8x_Up)
        # res4x_out = real_b_4x- res4x_out_I

        # res4x_out = self.convd4x_out(res4x_shared)
        # res4x_out_T = self.convd4x_out_T(res4x_shared)

        res4x_FI = F.interpolate(res4x_FI, encoder_layer[-3].size()[2:], mode='bilinear')
        res4x_Up = F.interpolate(res4x_out, encoder_layer[-3].size()[2:], mode='bilinear')
        res4x_FT = F.interpolate(res4x_FT, encoder_layer[-3].size()[2:], mode='bilinear')
        cl_2 = self.res_blk13(encoder_layer[-3])
        cl_21 = self.res_blk14(cl_2)
        res2x_I = self.conv2x_I(self.relu(cl_21))
        res2x_T = self.conv2x_T(self.relu(cl_21))

        res2x_I = torch.cat([res4x_FI, res2x_I], dim=1)
        res2x_I = self.convd4x_DI(self.relu(res2x_I))

        res2x_T = torch.cat([res4x_FT, res2x_T], dim=1)
        res2x_T = self.convd4x_DT(self.relu(res2x_T))

        # res2x_Ta = res2x_I + res2x_T
        res2x_I1 = self.pa2x_c_t(res2x_T) * res2x_I
        res2x_FI = self.pa2x_s_t(res2x_I1) * res2x_I1+res2x_I

        res2x_I2 = self.pa2x_c_m(res2x_I) * res2x_T
        res2x_FT = self.pa2x_s_m(res2x_I2) * res2x_I2+res2x_T
        # res2x_I = self.conv2x_11(torch.cat([res2x_I2, res2x_I1], dim=1))

        res2x_out_T =  self.convd2x_out_T(self.relu(res2x_FT))

        # real_b_2x = F.interpolate(in_x, scale_factor=1 / 2, mode='bilinear')
        # res2x_out_M = 2 * (real_b_2x + 1) / (res2x_out_T + 1 + (10 ** -10)) - (res2x_out_T + 1) - 1
        # res2x_out_M = (real_b_2x - 1) / (res2x_out_T + 1 + (10 ** -10)) + 1
        # res2x_out_M = res2x_out_M *2-1
        res2x_out_I = self.convd2x_out(self.relu(res2x_FI))
        res2x_out = (1 - self.weight3) *res2x_out_I  + self.weight3* res4x_Up
        # res2x_out =real_b_2x- res2x_out_I

        res2x_FI = F.interpolate(res2x_FI, encoder_layer[0].size()[2:], mode='bilinear')
        res2x_Up = F.interpolate(res2x_out, encoder_layer[0].size()[2:], mode='bilinear')
        res2x_FT = F.interpolate(res2x_FT, encoder_layer[0].size()[2:], mode='bilinear')
        cl_1 = self.res_blk15(encoder_layer[0])
        # resx_out = self.convdx_out(cl_1)
        resx_I = self.convx_I(self.relu(cl_1))
        resx_T = self.convx_T(self.relu(cl_1))

        resx_I = torch.cat([resx_I, res2x_FI], dim=1)
        resx_I = self.convd2x_DI(self.relu(resx_I))

        resx_T = torch.cat([resx_T, res2x_FT], dim=1)
        resx_T = self.convd2x_DT(self.relu(resx_T))

        # resx_Ta = resx_I + resx_T
        resx_I1 = self.pax_c_t(resx_T) * resx_I
        resx_FI = self.pax_s_t(resx_I1) * resx_I1+resx_I

        resx_I2 = self.pax_c_m(resx_I) * resx_T
        resx_FT = self.pax_s_m(resx_I2) * resx_I2+resx_T

        # resx_Ta = torch.cat([resx_I, resx_T], dim=1)
        # resx_I = self.pax(resx_Ta) * resx_I + resx_I
        out_T = self.convd_T(self.relu(resx_FT))

        # out_M = 2 * (in_x + 1) / (out_T + 1 + (10 ** -10)) - (out_T + 1) - 1

        # out_M = (in_x-1)/(out_T + 1 + (10 ** -10))+1
        # out_M = out_M*2-1
        out_I = self.conv_last(self.relu(resx_FI))
        out = (1 - self.weight4) * out_I + self.weight4* res2x_Up

        return F.tanh(out), F.tanh(out_T), F.tanh(res8x_out), F.tanh(res8x_out_T), F.tanh(res4x_out), F.tanh(res4x_out_T), F.tanh(res2x_out), F.tanh(res2x_out_T)

class Encoder_NewBlock_fusion_with_T625_r(nn.Module):
    def __init__(self, channel=64, conv=default_conv, res_blokcs=8):
        super(Encoder_NewBlock_fusion_with_T625_r, self).__init__()
        self.channels = channel
        self.conv_d1 = nn.Conv2d(3, self.channels, 7, 1, 3)
        
        self.conv_d2 = nn.Conv2d(self.channels, self.channels * 2, 4, 2, 1)
       
        self.conv_d3 = nn.Conv2d(self.channels * 2, self.channels * 4, 4, 2, 1)
       
        self.conv_d4 = nn.Conv2d(self.channels * 4, self.channels * 8, 4, 2, 1)
        
        self.res_blk0 = ResBlock(default_conv, self.channels * 8, 3, act=nn.ReLU())
        self.res_blk1 = ResBlock(default_conv, self.channels * 8, 3, act=nn.ReLU())
        self.res_blk2 = ResBlock(default_conv, self.channels * 8, 3, act=nn.ReLU())

        self.res_blk3 = ResBlock(default_conv, self.channels * 8, 3, act=nn.ReLU())
        self.res_blk4 = ResBlock(default_conv, self.channels * 8, 3, act=nn.ReLU())
        self.res_blk5 = ResBlock(default_conv, self.channels * 8, 3, act=nn.ReLU())

        self.res_blk6 = ResBlock(default_conv, self.channels * 8, 3, act=nn.ReLU())
        self.res_blk7 = ResBlock(default_conv, self.channels * 8, 3, act=nn.ReLU())
        self.res_blk8 = ResBlock(default_conv, self.channels * 8, 3, act=nn.ReLU())

        self.res_blk9 = ResBlock(default_conv, self.channels * 4, 3, act=nn.ReLU())
        self.res_blk10 = ResBlock(default_conv, self.channels * 4, 3, act=nn.ReLU())

        self.res_blk11 = ResBlock(default_conv, self.channels * 4, 3, act=nn.ReLU())
        self.res_blk12 = ResBlock(default_conv, self.channels * 4, 3, act=nn.ReLU())

        self.res_blk13 = ResBlock(default_conv, self.channels * 2, 3, act=nn.ReLU())
        self.res_blk14 = ResBlock(default_conv, self.channels * 2, 3, act=nn.ReLU())

        self.res_blk15 = ResBlock(default_conv, self.channels, 3, act=nn.ReLU())

        self.relu = nn.ReLU()
        self.conv1X11 = nn.Conv2d(self.channels * 32, self.channels * 8, 1, 1, 0)

        self.pa8x_s_m = ChannelAttention(self.channels*8)
        self.pa8x_s_t = ChannelAttention(self.channels * 8)
        self.pa8x_c_m = nn.Sequential(
            nn.Conv2d(self.channels * 8, self.channels * 8, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.channels * 8, self.channels * 8, 3, 1, 1),
            nn.Sigmoid()
        )
        self.pa8x_c_t = nn.Sequential(
            nn.Conv2d(self.channels * 8, self.channels * 8, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.channels * 8, self.channels * 8, 3, 1, 1),
            nn.Sigmoid()
        )
        
        self.conv8x_I = ResBlock(default_conv, self.channels * 8, 3, act=nn.ReLU())
        self.conv8x_T = ResBlock(default_conv, self.channels * 8, 3, act=nn.ReLU())

        self.convd8x_out = nn.Conv2d(self.channels * 8, 3, 3, 1, 1)
        self.convd8x_out_T = nn.Conv2d(self.channels * 8, 1, 3, 1, 1)
        self.convd8x_DI = nn.Sequential(
            nn.Conv2d(self.channels * 12, self.channels * 4, 1, 1, 0),
            nn.Conv2d(self.channels * 4, self.channels * 4, 3, 1, 1),
        )
        self.convd8x_DT = nn.Sequential(
            nn.Conv2d(self.channels * 12, self.channels * 4, 1, 1, 0),
            nn.Conv2d(self.channels * 4, self.channels * 4, 3, 1, 1),
        )

        self.conv4x_I = ResBlock(default_conv, self.channels * 4, 3, act=nn.ReLU())
        self.conv4x_T = ResBlock(default_conv, self.channels * 4, 3, act=nn.ReLU())
        self.convd4x_out = nn.Conv2d(self.channels * 4, 3, 3, 1, 1)
        self.convd4x_out_T = nn.Conv2d(self.channels * 4, 1, 3, 1, 1)
        
        self.convd4x_DI = nn.Sequential(
            nn.Conv2d(self.channels * 6, self.channels * 2, 1, 1, 0),
            nn.Conv2d(self.channels * 2, self.channels * 2, 3, 1, 1),
        )
        self.convd4x_DT = nn.Sequential(
            nn.Conv2d(self.channels * 6, self.channels * 2, 1, 1, 0),
            nn.Conv2d(self.channels * 2, self.channels * 2, 3, 1, 1),
        )

        self.pa4x_s_m = ChannelAttention(self.channels * 4)
        self.pa4x_s_t = ChannelAttention(self.channels * 4)
        self.pa4x_c_m = nn.Sequential(
            nn.Conv2d(self.channels * 4, self.channels * 4, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.channels * 4, self.channels * 4, 3, 1, 1),
            nn.Sigmoid()
        )
        self.pa4x_c_t = nn.Sequential(
            nn.Conv2d(self.channels * 4, self.channels * 4, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.channels * 4, self.channels * 4, 3, 1, 1),
            nn.Sigmoid()
        )
        

        self.conv2x_I = ResBlock(default_conv, self.channels * 2, 3, act=nn.ReLU())
        self.conv2x_T = ResBlock(default_conv, self.channels * 2, 3, act=nn.ReLU())
        self.convd2x_out = nn.Conv2d(self.channels * 2, 3, 3, 1, 1)
        self.convd2x_out_T = nn.Conv2d(self.channels * 2, 1, 3, 1, 1)
        self.convd2x_DI = nn.Sequential(
            nn.Conv2d(self.channels * 3, self.channels, 1, 1, 0),
            nn.Conv2d(self.channels, self.channels, 3, 1, 1),
        )
        self.convd2x_DT = nn.Sequential(
            nn.Conv2d(self.channels * 3, self.channels, 1, 1, 0),
            nn.Conv2d(self.channels, self.channels, 3, 1, 1),
        )

        self.pa2x_s_m = ChannelAttention(self.channels*2 )
        self.pa2x_s_t = ChannelAttention(self.channels*2 )
        self.pa2x_c_m =nn.Sequential(
            nn.Conv2d(self.channels * 2, self.channels * 2, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.channels * 2, self.channels * 2, 3, 1, 1),
            nn.Sigmoid()
        )
        self.pa2x_c_t = nn.Sequential(
            nn.Conv2d(self.channels * 2, self.channels * 2, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.channels * 2, self.channels * 2, 3, 1, 1),
            nn.Sigmoid()
        )
       
        self.convx_I = ResBlock(default_conv, self.channels, 3, act=nn.ReLU())
        self.convx_T = ResBlock(default_conv, self.channels, 3, act=nn.ReLU())
       
        self.pax_s_m = ChannelAttention(self.channels )
        self.pax_s_t = ChannelAttention(self.channels )
        self.pax_c_m = nn.Sequential(
            nn.Conv2d(self.channels , self.channels , 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.channels , self.channels , 3, 1, 1),
            nn.Sigmoid()
        )
        self.pax_c_t = nn.Sequential(
            nn.Conv2d(self.channels , self.channels , 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.channels , self.channels , 3, 1, 1),
            nn.Sigmoid()
        )
        
        self.conv_last = ConvLayer(self.channels, 3, kernel_size=7, stride=1)
        self.convd_T = ConvLayer(self.channels, 1, kernel_size=7, stride=1)

        self.weight1 = nn.Parameter(torch.Tensor([0]))
        self.weight2 = nn.Parameter(torch.Tensor([0]))
        self.weight3 = nn.Parameter(torch.Tensor([0]))
        self.weight4 = nn.Parameter(torch.Tensor([0]))

    def forward(self, in_x):
        encoder_layer = []
        x = self.conv_d1(in_x)
       
        encoder_layer.append(x)
        x = self.conv_d2(self.relu(x))
        
        encoder_layer.append(x)
        x = self.conv_d3(self.relu(x))
        
        encoder_layer.append(x)
        x = self.conv_d4(self.relu(x))
        encoder_layer.append(x)
        N, C, H, W = x.size()
        x0 = self.res_blk0(x)
        x1 = self.res_blk1(x0)
        x2 = self.res_blk2(x1)
        x3 = self.res_blk3(x2)
        x4 = self.res_blk4(x3)
        x5 = self.res_blk5(x4)
        x6 = self.res_blk6(x5)
        x7 = self.res_blk7(x6)
        x8 = self.res_blk8(x7)

        res8x = self.conv1X11(self.relu(torch.cat([x, x8, x2, x5], dim=1)))
        res8x_I = self.conv8x_I(self.relu(res8x))

        res8x_T = self.conv8x_T(self.relu(res8x))

       
        res8x_I1 = self.pa8x_c_t(res8x_T) * res8x_I
        res8x_FI = self.pa8x_s_t(res8x_I1) * res8x_I1
        res8x_FI = res8x_FI+res8x_I

        res8x_I2 = self.pa8x_c_m(res8x_I) * res8x_T
        res8x_FT = self.pa8x_s_m(res8x_I2) * res8x_I2
        res8x_FT = res8x_FT + res8x_T

        
        res8x_out_T = self.convd8x_out_T(self.relu(res8x_FT))

        
        res8x_out = self.convd8x_out(self.relu(res8x_FI))


        res8x_FI = F.interpolate(res8x_FI, encoder_layer[-2].size()[2:], mode='bilinear')
        res8x_Up = F.interpolate(res8x_out, encoder_layer[-2].size()[2:], mode='bilinear')
        res8x_FT = F.interpolate(res8x_FT, encoder_layer[-2].size()[2:], mode='bilinear')
        cl_4 = self.res_blk9(encoder_layer[-2])
        cl_41 = self.res_blk10(cl_4)
        cl_43 = self.res_blk11(cl_41)
        cl_44 = self.res_blk12(cl_43)
        res4x_I = self.conv4x_I(self.relu(cl_44))
        res4x_T = self.conv4x_T(self.relu(cl_44))

        res4x_I = torch.cat([res8x_FI, res4x_I], dim=1)
        res4x_I = self.convd8x_DI(self.relu(res4x_I))

        res4x_T = torch.cat([res8x_FT, res4x_T], dim=1)
        res4x_T = self.convd8x_DT(self.relu(res4x_T))

        # res4x_Ta = res4x_I + res4x_T
        res4x_I1 = self.pa4x_c_t(res4x_T) * res4x_I
        res4x_FI = self.pa4x_s_t(res4x_I1) * res4x_I1+res4x_I

        res4x_I2 = self.pa4x_c_m(res4x_I) * res4x_T
        res4x_FT = self.pa4x_s_m(res4x_I2) * res4x_I2+res4x_T
        

        res4x_out_T =  self.convd4x_out_T(self.relu(res4x_FT))

        
        res4x_out_I = self.convd4x_out(self.relu(res4x_FI))
        res4x_out = (1 - self.weight2) * res4x_out_I + self.weight2 * (res8x_Up)
        

        res4x_FI = F.interpolate(res4x_FI, encoder_layer[-3].size()[2:], mode='bilinear')
        res4x_Up = F.interpolate(res4x_out, encoder_layer[-3].size()[2:], mode='bilinear')
        res4x_FT = F.interpolate(res4x_FT, encoder_layer[-3].size()[2:], mode='bilinear')
        cl_2 = self.res_blk13(encoder_layer[-3])
        cl_21 = self.res_blk14(cl_2)
        res2x_I = self.conv2x_I(self.relu(cl_21))
        res2x_T = self.conv2x_T(self.relu(cl_21))

        res2x_I = torch.cat([res4x_FI, res2x_I], dim=1)
        res2x_I = self.convd4x_DI(self.relu(res2x_I))

        res2x_T = torch.cat([res4x_FT, res2x_T], dim=1)
        res2x_T = self.convd4x_DT(self.relu(res2x_T))

        # res2x_Ta = res2x_I + res2x_T
        res2x_I1 = self.pa2x_c_t(res2x_T) * res2x_I
        res2x_FI = self.pa2x_s_t(res2x_I1) * res2x_I1+res2x_I

        res2x_I2 = self.pa2x_c_m(res2x_I) * res2x_T
        res2x_FT = self.pa2x_s_m(res2x_I2) * res2x_I2+res2x_T
        # res2x_I = self.conv2x_11(torch.cat([res2x_I2, res2x_I1], dim=1))

        res2x_out_T =  self.convd2x_out_T(self.relu(res2x_FT))

        
        res2x_out_I = self.convd2x_out(self.relu(res2x_FI))
        res2x_out = (1 - self.weight3) *res2x_out_I  + self.weight3* res4x_Up
       

        res2x_FI = F.interpolate(res2x_FI, encoder_layer[0].size()[2:], mode='bilinear')
        res2x_Up = F.interpolate(res2x_out, encoder_layer[0].size()[2:], mode='bilinear')
        res2x_FT = F.interpolate(res2x_FT, encoder_layer[0].size()[2:], mode='bilinear')
        cl_1 = self.res_blk15(encoder_layer[0])
       
        resx_I = self.convx_I(self.relu(cl_1))
        resx_T = self.convx_T(self.relu(cl_1))

        resx_I = torch.cat([resx_I, res2x_FI], dim=1)
        resx_I = self.convd2x_DI(self.relu(resx_I))

        resx_T = torch.cat([resx_T, res2x_FT], dim=1)
        resx_T = self.convd2x_DT(self.relu(resx_T))

        # resx_Ta = resx_I + resx_T
        resx_I1 = self.pax_c_t(resx_T) * resx_I
        resx_FI = self.pax_s_t(resx_I1) * resx_I1+resx_I

        resx_I2 = self.pax_c_m(resx_I) * resx_T
        resx_FT = self.pax_s_m(resx_I2) * resx_I2+resx_T

        
        out_T = self.convd_T(self.relu(resx_FT))

        
        out_I = self.conv_last(self.relu(resx_FI))
        out = (1 - self.weight4) * out_I + self.weight4* res2x_Up

        return F.tanh(out), F.tanh(out_T), F.tanh(res8x_out), F.tanh(res8x_out_T), F.tanh(res4x_out), F.tanh(res4x_out_T), F.tanh(res2x_out), F.tanh(res2x_out_T)



