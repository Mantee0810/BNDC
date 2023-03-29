import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torchvision
import ptflops

from torch.nn import BatchNorm2d


class ConvBNReLU(nn.Module):
    def __init__(self, channel, ks=(3, 3), stride=(1, 1), padding=(1, 1), *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(channel,
                              channel,
                              kernel_size=ks,
                              stride=stride,
                              padding=padding,
                              bias=False,
                              groups=channel)
        self.bn = BatchNorm2d(channel)
        self.relu = nn.ReLU(inplace=True)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class Inter_channel(nn.Module):

    def __init__(self, channel):
        super(Inter_channel, self).__init__()
        self.channel = channel
        self.conv1 = None
        self.conv2 = None
        self.linear1 = nn.Linear(channel, 2 * channel)
        self.linear2 = nn.Linear(2 * channel, channel)
        self.sm = nn.Softmax(dim=1)

    def forward(self, x):
        b, c, h, w = x.shape
        stride = (1,1)
        if int(h/w) == 2:
            stride1 = stride
            stride2 = (2,1)
        elif int(h/w) == 1:
            stride1 = stride
            stride2 = stride
        elif int(w/h) == 2:
            stride1 = (1,2)
            stride2 = stride

        # 调整这两个stride可以影响输入输出大小
        if self.conv1 is None:
            self.conv1 = ConvBNReLU(self.channel, ks=(h, 1), stride=stride1, padding=(0, 0))
        if self.conv2 is None:
            self.conv2 = ConvBNReLU(self.channel, ks=(1, w), stride=stride2, padding=(0, 0))

        xl1 = self.conv1(x)
        xl2 = self.conv2(x)

        # len of res1 and res2 is channel
        res1 = []
        res2 = []
        num = []
        for i in range(c):
            res1.append(xl1[:, i, :, :])
            res2.append(xl2[:, i, :, :])

        for j in range(c):
            dot1 = res1[j]
            dot2 = res2[j]
            ans = []
            for k in range(b):
                cul1 = dot1[k, :, :]
                cul2 = dot2[k, :, :]
                ans.append(torch.mm(cul1, cul2))
            # ans pinjie
            if b == 1:
                ans = ans[0]
            else:
                for l in range(1, len(ans)):
                    ans = torch.cat((ans[0], ans[l]))
            num.append(ans)

        # num to tensor
        num = torch.stack(num, dim=0)
        num = num.view(b, c)
        num = self.linear1(num)
        num = self.linear2(num)
        num = self.sm(num)
        num = num.view(b, c, 1, 1)

        return x * num.expand_as(x)


class Intra_channel(nn.Module):
    def __init__(self, channels, dimension=2, sub_sample=False, bn_layer=True):

        super(Intra_channel, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = channels
        self.inter_channels = channels

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0, groups=self.in_channels)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0, groups=self.in_channels),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0, groups=self.in_channels)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0, groups=self.in_channels)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0, groups=self.in_channels)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x, return_nl_map=False):
        """
        :param x: (b, c, t, h, w)
        :param return_nl_map: if True return z, nl_map, else only return z.
        :return:
        """

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        if return_nl_map:
            return z, f_div_C
        return z


if __name__ == "__main__":
    input1 = torch.randn(size=(2, 128, 16,16))

    ic = Inter_channel(128)
    nl = Intra_channel(128)
    output1 = ic(input1)
    output2 = nl(input1)
    print(output1.shape,output2.shape)
