from __future__ import print_function, division
import torch.nn as nn
import torch.utils.data
import torch
import numpy as np

filters = [16, 32, 48]

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 2, kernel_size, padding=kernel_size // 2, bias=False)
        self.conv2 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return self.sigmoid(x)

class Ca(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(Ca, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        attn = self.softmax(out)
        return attn

class Sobelxy(nn.Module):
    def __init__(self, channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(Sobelxy, self).__init__()
        sobel_filter = np.array([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])
        self.convx = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=channels, bias=False)
        self.convx.weight.data.copy_(torch.from_numpy(sobel_filter))
        self.convy=nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=channels, bias=False)
        self.convy.weight.data.copy_(torch.from_numpy(sobel_filter.T))
    def forward(self, x):
        sobelx = self.convx(x)
        sobely = self.convy(x)
        x = torch.abs(sobelx) + torch.abs(sobely)
        return x

class MSDR(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(MSDR, self).__init__()

        self.sobel = Sobelxy(out_ch)

        self.convlayer1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, groups=1),
            nn.BatchNorm2d(out_ch),
            nn.Tanh(),
        )
        self.convlayer2 = nn.Sequential(
            nn.Conv2d(9*out_ch, out_ch, kernel_size=1, stride=1, padding=0, groups=1),
            nn.BatchNorm2d(out_ch),
            nn.Tanh(),
            nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1, padding=0, groups=1),
            nn.BatchNorm2d(out_ch),
            nn.Tanh(),
        )

        self.con3x3_1 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, groups=1),
            nn.BatchNorm2d(out_ch),
            nn.Tanh(),
        )
        self.con3x3_2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, groups=1),
            nn.BatchNorm2d(out_ch),
            nn.Tanh(),
        )
        self.con3x3_3 = nn.Sequential(
            nn.Conv2d(2*out_ch, out_ch, kernel_size=3, stride=1, padding=1, groups=1),
            nn.BatchNorm2d(out_ch),
            nn.Tanh(),
        )

        self.con5x5_1 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=2, groups=1),
            nn.BatchNorm2d(out_ch),
            nn.Tanh(),
        )
        self.con5x5_2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=2, groups=1),
            nn.BatchNorm2d(out_ch),
            nn.Tanh(),
        )
        self.con5x5_3 = nn.Sequential(
            nn.Conv2d(2 * out_ch, out_ch, kernel_size=5, stride=1, padding=2, groups=1),
            nn.BatchNorm2d(out_ch),
            nn.Tanh(),
        )

        self.con7x7_1 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=7, stride=1, padding=3, groups=1),
            nn.BatchNorm2d(out_ch),
            nn.Tanh(),
        )

        self.con7x7_2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=7, stride=1, padding=3, groups=1),
            nn.BatchNorm2d(out_ch),
            nn.Tanh(),
        )
        self.con7x7_3 = nn.Sequential(
            nn.Conv2d(2 * out_ch, out_ch, kernel_size=7, stride=1, padding=3, groups=1),
            nn.BatchNorm2d(out_ch),
            nn.Tanh(),
        )

    def forward(self, x):
        fe_conv = self.convlayer1(x)
        edge = self.sobel(fe_conv)

        fe_3x3_1 = self.con3x3_1(fe_conv)
        fe_3x3_2 = self.con3x3_2(fe_3x3_1)
        fe_3x3_3 = self.con3x3_3(torch.cat((fe_3x3_1, fe_3x3_2), dim=1))

        fe_5x5_1 = self.con5x5_1(fe_conv)
        fe_5x5_2 = self.con5x5_2(fe_5x5_1)
        fe_5x5_3 = self.con5x5_3(torch.cat((fe_5x5_1, fe_5x5_2), dim=1))

        fe_7x7_1 = self.con7x7_1(fe_conv)
        fe_7x7_2 = self.con7x7_2(fe_7x7_1)
        fe_7x7_3 = self.con7x7_3(torch.cat((fe_7x7_1, fe_7x7_2), dim=1))

        out_1 = self.convlayer2(torch.cat((fe_3x3_1, fe_3x3_2, fe_3x3_3, fe_5x5_1, fe_5x5_2, fe_5x5_3, fe_7x7_1,
                                         fe_7x7_2, fe_7x7_3), dim=1))
        out = out_1 + fe_conv + edge

        return out

class CMIA(nn.Module):
    """
    Convolution Block
    """
    def __init__(self, in_ch):
        super(CMIA, self).__init__()
        self.sa = SpatialAttention()
        self.ca = Ca(in_ch)
        self.sigmoid = nn.Sigmoid()
        self.conv = nn.Conv2d(2*in_ch, in_ch, kernel_size=1, padding=0, stride=1)
        self.conv1 = nn.Conv2d(3*in_ch, 1, kernel_size=1, padding=0, stride=1)

    def forward(self, vis, ir):
        fe = self.conv(torch.cat((ir, vis), dim=1))
        fe_sa = self.sa(fe)
        fe_ca = self.ca(fe)
        attn = self.sigmoid(self.conv1(torch.cat((fe_ca*fe, fe, fe_sa*fe), dim=1)))
        ir = ir * attn
        vis = vis * (1 - attn)

        return vis, ir

class ConvRelu2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super(ConvRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

class ConvBnTanh2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvBnTanh2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        self.bn   = nn.BatchNorm2d(out_channels)
    def forward(self,x):
        return torch.tanh(self.conv(x))/2+0.5

class Resblock(nn.Module):
    """
    Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(Resblock, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False)
        self.conv1 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x):
        res = x
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = x + res

        return x

class Encoder(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """
    def __init__(self, in_ch=1):
        super(Encoder, self).__init__()
        self.Layer1_vis = nn.Sequential(
            ConvRelu2d(in_ch, filters[1]),
            ConvRelu2d(filters[1], filters[2]),
            MSDR(filters[2], filters[2]),
        )
        self.Layer2_vis = nn.Sequential(
            ConvRelu2d(filters[2], filters[2]),
            ConvRelu2d(filters[2], filters[2]),
            MSDR(filters[2], filters[2]),
        )
        self.Layer3_vis = nn.Sequential(
            ConvRelu2d(2*filters[2], filters[2]),
            ConvRelu2d(filters[2], filters[2]),
            MSDR(filters[2], filters[2]),
        )
        self.Layer4_vis = nn.Sequential(
            ConvRelu2d(3*filters[2], filters[2]),
            ConvRelu2d(filters[2], filters[2]),
            MSDR(filters[2], filters[2]),
        )

        self.Layer1_ir = nn.Sequential(
            ConvRelu2d(in_ch, filters[1]),
            ConvRelu2d(filters[1], filters[2]),
            MSDR(filters[2], filters[2]),
        )
        self.Layer2_ir = nn.Sequential(
            ConvRelu2d(filters[2], filters[2]),
            ConvRelu2d(filters[2], filters[2]),
            MSDR(filters[2], filters[2]),
        )
        self.Layer3_ir = nn.Sequential(
            ConvRelu2d(2*filters[2], filters[2]),
            ConvRelu2d(filters[2], filters[2]),
            MSDR(filters[2], filters[2]),
        )
        self.Layer4_ir = nn.Sequential(
            ConvRelu2d(3*filters[2], filters[2]),
            ConvRelu2d(filters[2], filters[2]),
            MSDR(filters[2], filters[2]),
        )

    def forward(self, vis, ir):
        vis_1 = self.Layer1_vis(vis)
        ir_1 = self.Layer1_ir(ir)

        vis_2 = self.Layer2_vis(vis_1)
        ir_2 = self.Layer2_ir(ir_1)

        vis_3 = self.Layer3_vis(torch.cat((vis_2, vis_1), dim=1))
        ir_3 = self.Layer3_ir(torch.cat((ir_2, ir_1), dim=1))

        out_vis = self.Layer4_vis(torch.cat((vis_1, vis_2, vis_3), dim=1))
        out_ir = self.Layer4_ir(torch.cat((ir_1, ir_2, ir_3), dim=1))

        return vis_1, vis_2, vis_3, out_vis, ir_1, ir_2, ir_3, out_ir

class Decoder_fu(nn.Module):
    def __init__(self, out_ch=1):
        super(Decoder_fu, self).__init__()
        self.conv1 = nn.Sequential(
            ConvRelu2d(in_channels=2*filters[2], out_channels=2*filters[2], kernel_size=3, stride=1, padding=1),
            ConvRelu2d(in_channels=2*filters[2], out_channels=filters[2], kernel_size=3, stride=1, padding=1),
        )
        self.RB1 = nn.Sequential(
            Resblock(filters[2], filters[2]),
            Resblock(filters[2], filters[2]),
        )
        self.conv2 = ConvRelu2d(in_channels=filters[2], out_channels=filters[1], kernel_size=3, stride=1, padding=1)
        self.RB2 = nn.Sequential(
            Resblock(filters[1], filters[1]),
            Resblock(filters[1], filters[1]),
        )
        self.conv3 = ConvRelu2d(in_channels=filters[1], out_channels=filters[0], kernel_size=3, stride=1, padding=1)
        self.RB3 = nn.Sequential(
            Resblock(filters[0], filters[0]),
            Resblock(filters[0], filters[0]),
        )
        self.conv4 = ConvRelu2d(in_channels=filters[0], out_channels=out_ch, kernel_size=3, stride=1, padding=1)
        self.conv5 = ConvBnTanh2d(in_channels=out_ch, out_channels=out_ch, kernel_size=1, stride=1, padding=0)

        self.Decoder_ir = ConvBnTanh2d(in_channels=filters[2], out_channels=out_ch, kernel_size=1,  padding=0, stride=1)
        self.Decoder_vi = ConvBnTanh2d(in_channels=filters[2], out_channels=out_ch, kernel_size=1,  padding=0, stride=1)

    def forward(self, vis, ir):
        x_cat = self.conv1(torch.cat((vis, ir), dim=1))
        x_1 = self.RB1(x_cat)
        x_2 = self.conv2(x_1)
        x_2 = self.RB2(x_2)
        x_3 = self.conv3(x_2)
        x_3 = self.RB3(x_3)
        out = self.conv4(x_3)
        re_ir = self.Decoder_ir(x_cat)
        re_vi = self.Decoder_vi(x_cat)
        return out, re_ir, re_vi

class Decoder_sg(nn.Module):
    def __init__(self, n_classes=9):
        super(Decoder_sg, self).__init__()
        self.conv1 = nn.Sequential(
            ConvRelu2d(in_channels=2 * filters[2], out_channels=2 * filters[2], kernel_size=3, stride=1, padding=1),
            ConvRelu2d(in_channels=2 * filters[2], out_channels=filters[2], kernel_size=3, stride=1, padding=1),
        )
        self.RB1 = nn.Sequential(
            Resblock(filters[2], filters[2]),
            Resblock(filters[2], filters[2]),
        )
        self.conv2 = ConvRelu2d(in_channels=filters[2], out_channels=filters[1], kernel_size=3, stride=1, padding=1)
        self.RB2 = nn.Sequential(
            Resblock(filters[1], filters[1]),
            Resblock(filters[1], filters[1]),
        )
        self.conv3 = ConvRelu2d(in_channels=filters[1], out_channels=filters[0], kernel_size=3, stride=1, padding=1)
        self.RB3 = nn.Sequential(
            Resblock(filters[0], filters[0]),
            Resblock(filters[0], filters[0]),
        )
        self.conv4 = ConvRelu2d(in_channels=filters[0], out_channels=n_classes, kernel_size=1, stride=1, padding=0)
        self.conv5 = nn.Conv2d(in_channels=n_classes, out_channels=n_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, vis, ir):
        x = self.conv1(torch.cat((vis, ir), dim=1))
        x = self.RB1(x)
        x = self.conv2(x)
        x = self.RB2(x)
        x = self.conv3(x)
        x = self.RB3(x)
        x = self.conv4(x)
        out = self.conv5(x)
        return out

class Decoder(nn.Module):
    def __init__(self, out_ch=1, n_classes=9):
        super(Decoder, self).__init__()
        self.de_fu = Decoder_fu(out_ch)
        self.de_sg = Decoder_sg(n_classes)

    def forward(self, vis_de, ir_de):
        fuse_out, re_ir, re_vi = self.de_fu(vis_de, ir_de)
        seg_out = self.de_sg(vis_de, ir_de)
        return fuse_out, re_ir, re_vi, seg_out