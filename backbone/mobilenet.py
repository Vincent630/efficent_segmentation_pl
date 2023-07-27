from torch import nn
import torch
from typing import Any, List


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

class Backbone_Net(nn.Module):
    def __init__(self):
        super(Backbone_Net,self).__init__()
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

    def forward(self,x):
        x = self.stem(x)
        c2 = self.stage1(x)
        out = self.stage2(c2) # 1/8 downsample
        return out
if __name__ == "__main__":
    input = torch.rand(1,3,224,416)
    net = Backbone_Net()
    net.eval()
    output = net(input)
    print(output.shape)
    torch.onnx.export(net,input,"mobilenet.onnx",input_names = ["input"],\
        output_names = ["output"],opset_version = 12)
    
        

    