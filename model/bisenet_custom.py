import torch
from torch import nn
from typing import Any, List
import math
from torch.cuda.amp import autocast
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.sgd import SGD
import importlib
import sys
import os
sys.path.append("../")



class CBR(nn.Module):
    def __init__(self, inp, oup, kernel_size=3, stride=1, groups=1, dilation=1):
        super(CBR, self).__init__()
        padding = (kernel_size - 1) // 2 * dilation
        self.layers = nn.Sequential(
            nn.Conv2d(inp, oup,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      dilation=dilation,
                      groups=groups,
                      bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU6(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)


class InvertBlock(nn.Module):
    def __init__(self, inp, oup, expand_ratio, kernel_size=3, stride=1):
        super(InvertBlock, self).__init__()

        inner_channel = int(inp * expand_ratio)

        self.res_connect = stride == 1 and inp == oup

        layers: List[nn.Module] = list()

        if expand_ratio != 1:
            layers.append(
                CBR(inp, inner_channel, 1, 1)
            )
        layers.extend([
            CBR(inner_channel, inner_channel, kernel_size=kernel_size, stride=stride, groups=inner_channel),
            nn.Conv2d(inner_channel, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup)
        ])
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        if self.res_connect:
            return x + self.layer(x)
        else:
            return self.layer(x)


class AttentionRefineModule(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(AttentionRefineModule, self).__init__()
        self.conv = InvertBlock(in_channel, out_channel, 1, 3, 1)
        self.conv_attn = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Sigmoid()
        )
        self.avg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        feat = self.conv(x)
        attn = self.conv_attn(self.avg(feat))
        return feat * attn


class FeatureFusionModule(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(FeatureFusionModule, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )
        self.inner_conv = InvertBlock(out_channel, out_channel, 1, 3, 1)
        self.attn = nn.Sequential(
            nn.Conv2d(out_channel, out_channel // 4, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channel // 4),
            nn.ReLU6(inplace=True),
            nn.Conv2d(out_channel // 4, out_channel, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Sigmoid()
        )
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.is_fuse = False

    def forward(self, fsp, fcp):
        feat = self.conv(torch.cat([fsp, fcp], dim=1))
        feat = self.inner_conv(feat)
        attn = self.attn(self.avg(feat))
        return feat + feat * attn


class BiSeNetOutput(nn.Module):
    def __init__(self, in_channel, mid_channel, num_classes):
        super(BiSeNetOutput, self).__init__()
        self.conv = InvertBlock(in_channel, mid_channel, 2, 3, 1)
        self.head = nn.Conv2d(mid_channel, num_classes, 1, bias=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.head(x)
        return x


class BiSeNetOutputExt(nn.Module):
    def __init__(self, in_channel, mid_channel, scale_factor, num_classes):
        super(BiSeNetOutputExt, self).__init__()
        self.conv = InvertBlock(in_channel, mid_channel, 2, 3, 1)

        #self.up = nn.UpsamplingBilinear2d(scale_factor=scale_factor)
        self.up = nn.UpsamplingBilinear2d(scale_factor=scale_factor)
        self.head = nn.Conv2d(mid_channel, num_classes, 1, bias=True)
    def forward(self, x):

        x = self.conv(x)
        
        x = self.up(x)
        x = self.head(x)
        return x
class BiSeNetOutputExtPixelShuffle(nn.Module):
    def __init__(self, in_channel, mid_channel, scale_factor, num_classes):
        super(BiSeNetOutputExtPixelShuffle, self).__init__()
        self.conv = InvertBlock(in_channel, mid_channel, 2, 3, 1)

        self.head = nn.Conv2d(mid_channel, num_classes, 1, bias=True)
        self.pixel_shuffle_2x = nn.PixelShuffle(2)
        self.pixel_shuffle_4x = nn.PixelShuffle(4)
        self.convs_2x = nn.Conv2d(4, 16, 1, bias=True)
        self.convs_4x = nn.Conv2d(1, 16, 1, bias=True)
        self.scale_factor = scale_factor
    def forward(self, x):

        x = self.conv(x)
        x = self.pixel_shuffle_2x(x)
        x = self.convs_2x(x)
        x = self.pixel_shuffle_4x(x)
        x = self.convs_4x(x)
        
        x = self.head(x)
        return x

class BiSeNetOutputInterpolate(nn.Module):
    def __init__(self, in_channel, mid_channel, scale_factor, num_classes):
        super(BiSeNetOutputInterpolate, self).__init__()
        self.conv = InvertBlock(in_channel, mid_channel, 2, 3, 1)

        self.head = nn.Conv2d(mid_channel, num_classes, 1, bias=True)
        self.up = F.interpolate(scale_factor=scale_factor, mode='bilinear')

    def forward(self, x):

        x = self.conv(x)
        x = self.up(x)
        
        x = self.head(x)
        return x


class MobileBisenet(pl.LightningModule):
    def __init__(self,
                 num_classes=3,
                 inner_channel=48,
                 inter_c4=True,
                 inter_c5=True,
                 edge_out=True,
                 export=False,
                 bininear_upsample=True,
                 model_name = "mobilenet",
                 split_stage = True
                 ):
        super(MobileBisenet, self).__init__()
        self.export = export
        self.split_stage = split_stage
        self.backbone_8x_downsample =  importlib.import_module("backbone.{:s}".format(model_name)).Backbone_Net()
        self.stem = CBR(
            3, 16, 3, 2
        )

        # 16 24 32 64 96
        self.stage1 = nn.Sequential(
            InvertBlock(16, 24, 1, 5, 2),
            InvertBlock(24, 24, 2, 3, 1)
        )
        self.stage2 = nn.Sequential(
            InvertBlock(24, 32, 2, 5, 2),
            InvertBlock(32, 32, 2, 3, 1),
            InvertBlock(32, 32, 2, 3, 1),
            InvertBlock(32, 32, 2, 3, 1),
        )

        self.stage3 = nn.Sequential(
            InvertBlock(32, 64, 2, 5, 2),
            InvertBlock(64, 64, 2, 3, 1),
            InvertBlock(64, 64, 2, 3, 1),
            InvertBlock(64, 64, 2, 3, 1),
            InvertBlock(64, 64, 2, 3, 1),
            InvertBlock(64, 64, 2, 3, 1),
            InvertBlock(64, 64, 2, 3, 1),
            InvertBlock(64, 64, 2, 3, 1),
        )

        self.state4 = nn.Sequential(
            InvertBlock(64, 96, 2, 5, 2),
            InvertBlock(96, 96, 2, 3, 1),
            InvertBlock(96, 96, 2, 3, 1),
            InvertBlock(96, 96, 2, 3, 1),
        )
        c3, c4, c5 = 32, 64, 96
        inner_channel = inner_channel
        self.avg = nn.AdaptiveAvgPool2d(1)

        self.arm_c5 = AttentionRefineModule(c5, inner_channel)
        self.arm_c4 = AttentionRefineModule(c4, inner_channel)

        self.conv_avg = nn.Sequential(
            nn.Conv2d(c5, inner_channel, 1, 1, bias=False),
            nn.BatchNorm2d(inner_channel),
            nn.ReLU6(inplace=True)
        )
        self.head_c5 = InvertBlock(inner_channel, inner_channel, 2, 3, 1)
        self.head_c4 = InvertBlock(inner_channel, inner_channel, 2, 3, 1)
        self.ffm = FeatureFusionModule(c3 + inner_channel, inner_channel)

        if bininear_upsample:
            self.out = BiSeNetOutputExt(inner_channel, inner_channel, 8, num_classes)
        else:
            self.out = BiSeNetOutputExtPixelShuffle(inner_channel, inner_channel, 8, num_classes)

        self.out_c5_up = None
        self.out_c4_up = None
        self.edge_out = None

        if inter_c5:
            self.out_c5_up = BiSeNetOutputExt(inner_channel, inner_channel, 16, num_classes)
        if inter_c4:
            self.out_c4_up = BiSeNetOutputExt(inner_channel, inner_channel, 8, num_classes)
        if edge_out:
            self.edge_out = BiSeNetOutputExt(inner_channel, inner_channel, 8, num_classes)
        # self.train_mode = True
    
    def forward_train(self, x):

        c3 = None
        if self.split_stage:
            x = self.stem(x)
            c2 = self.stage1(x)
            c3 = self.stage2(c2)
        else:
            c3 = self.backbone_8x_downsample(x)
        c4 = self.stage3(c3)
        c5 = self.state4(c4)
        avg = self.conv_avg(self.avg(c5))
        arm_c5 = self.arm_c5(c5)
        c5_up = self.head_c5(nn.UpsamplingNearest2d(scale_factor=2)(arm_c5 + avg))
        arm_c4 = self.arm_c4(c4)
        c4_up = self.head_c4(nn.UpsamplingNearest2d(scale_factor=2)(arm_c4 + c5_up))
        feat_fuse = self.ffm(c3, c4_up)
        out = [None, None, None, None]
        feat_out = self.out(feat_fuse)
       
        out[0] = feat_out
        if self.out_c4_up:
            feat_c4_up = self.out_c4_up(c4_up)
            out[1] = feat_c4_up
        if self.out_c5_up:
            feat_c5_up = self.out_c5_up(c5_up)
            out[2] = feat_c5_up
        if self.edge_out:
            out[3] = self.edge_out(feat_fuse)

        return out

    def forward_val(self, x):
        if self.split_stage:
            x = self.stem(x)
            c2 = self.stage1(x)
            c3 = self.stage2(c2)
        else:
            c3 = self.backbone_8x_downsample(x)
        c4 = self.stage3(c3)
        c5 = self.state4(c4)

        avg = self.conv_avg(self.avg(c5))
        arm_c5 = self.arm_c5(c5)
        c5_up = self.head_c5(nn.UpsamplingNearest2d(scale_factor=2)(arm_c5 + avg))
        arm_c4 = self.arm_c4(c4)
        c4_up = self.head_c4(nn.UpsamplingNearest2d(scale_factor=2)(arm_c4 + c5_up))
        feat_fuse = self.ffm(c3, c4_up)
        feat_out = self.out(feat_fuse)

        if self.export:
            feat_out = feat_out.argmax(dim=1, keepdim=True)
        return feat_out

    def forward(self, x):
        if self.training:
            return self.forward_train(x)
        else:
            return self.forward_val(x)


if __name__ == '__main__':
    net = MobileBisenet(inner_channel=16)
    inp = torch.rand(size=(2, 3, 256, 448))
    net.eval()
    out = net(inp)
    print(out.shape)
    # print(net)
    input = torch.randn((1,3,640,480))
    torch.onnx.export(net,input,"bisenet_v2_custom.onnx",input_names = ["input"],\
        output_names = ["output"],opset_version = 12)
