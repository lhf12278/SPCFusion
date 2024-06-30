from __future__ import print_function, division
import torch.nn as nn
import torch.utils.data
import torch
import numpy as np

filters = [16, 32, 48]

def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d-mi)/(ma-mi)
    return dn

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
                                # nn.ReLU(),
                                nn.Sigmoid(),
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

class ConvRelu2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.Sigmoid()
        # self.relu = nn.ReLU()
    def forward(self,x):
        return self.relu(self.bn(self.conv(x)))

class MSRM(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(MSRM, self).__init__()

        self.sobel = Sobelxy(out_ch)

        self.convlayer1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.Sigmoid(),
            # nn.ReLU(),
        )
        self.convlayer2 = nn.Sequential(
            nn.Conv2d(9*out_ch, out_ch, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_ch),
            nn.Sigmoid(),
            # nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_ch),
            nn.Sigmoid(),
            # nn.ReLU(),
        )

        self.con3x3_1 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            # nn.ReLU(),
            nn.Sigmoid(),
        )
        self.con3x3_2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            # nn.ReLU(),
            nn.Sigmoid(),
        )
        self.con3x3_3 = nn.Sequential(
            nn.Conv2d(2*out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            # nn.ReLU(),
            nn.Sigmoid(),
        )

        self.con5x5_1 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(out_ch),
            # nn.ReLU(),
            nn.Sigmoid(),
        )
        self.con5x5_2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(out_ch),
            # nn.ReLU(),
            nn.Sigmoid(),
        )
        self.con5x5_3 = nn.Sequential(
            nn.Conv2d(2 * out_ch, out_ch, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(out_ch),
            # nn.ReLU(),
            nn.Sigmoid(),
        )

        self.con7x7_1 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(out_ch),
            # nn.ReLU(),
            nn.Sigmoid(),
        )

        self.con7x7_2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(out_ch),
            # nn.ReLU(),
            nn.Sigmoid(),
        )
        self.con7x7_3 = nn.Sequential(
            nn.Conv2d(2 * out_ch, out_ch, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(out_ch),
            nn.Sigmoid(),
            # nn.ReLU(),
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

class conv_block(nn.Module):
    """
    Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, stride=1)
        self.conv1 = ConvRelu2d(out_ch, out_ch)
        self.conv2 = ConvRelu2d(2*out_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        x = torch.cat((x, self.conv1(x)), dim=1)
        x = self.conv2(x)
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
            MSRM(filters[2], filters[2]),
            # MSRM(filters[2], filters[2]),
        )
        self.Layer2_vis = nn.Sequential(
            MSRM(filters[2], filters[2]),
            # MSRM(filters[2], filters[2])
        )
        self.Layer3_vis = nn.Sequential(
            MSRM(filters[2], filters[2]),
            # MSRM(filters[2], filters[2])
        )
        self.Conv_vis = nn.Sequential(
            ConvRelu2d(3*filters[2], filters[2], kernel_size=1, stride=1, padding=0)
        )

        self.Layer1_ir = nn.Sequential(
            ConvRelu2d(in_ch, filters[1]),
            ConvRelu2d(filters[1], filters[2]),
            MSRM(filters[2], filters[2]),
            # MSRM(filters[2], filters[2])
        )
        self.Layer2_ir = nn.Sequential(
            MSRM(filters[2], filters[2]),
            # MSRM(filters[2], filters[2])
        )
        self.Layer3_ir = nn.Sequential(
            MSRM(filters[2], filters[2]),
            # MSRM(filters[2], filters[2])
        )
        self.Conv_ir = nn.Sequential(
            ConvRelu2d(3*filters[2], filters[2], kernel_size=1, stride=1, padding=0)
        )


    def forward(self, vis, ir):
        vis_1 = self.Layer1_vis(vis)
        vis_2 = self.Layer2_vis(vis_1)
        vis_3 = self.Layer3_vis(vis_2)
        out_vis = self.Conv_vis(torch.cat((vis_1, vis_2, vis_3), dim=1))

        ir_1 = self.Layer1_ir(ir)
        ir_2 = self.Layer2_ir(ir_1)
        ir_3 = self.Layer3_ir(ir_2)
        out_ir = self.Conv_ir(torch.cat((ir_1, ir_2, ir_3), dim=1))

        return vis_1, vis_2, vis_3, out_vis, ir_1, ir_2, ir_3, out_ir

class Decoder_fu(nn.Module):
    def __init__(self, out_ch=1):
        super(Decoder_fu, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.sa_ir = SpatialAttention()
        self.sa_vi = SpatialAttention()
        self.ca_ir = Ca(filters[2])
        self.ca_vi = Ca(filters[2])
        self.fuse_ir = nn.Sequential(
            conv_block(2*filters[2], filters[2]),
            conv_block(filters[2], filters[2]),
        )
        self.fuse_vi = nn.Sequential(
            conv_block(2*filters[2], filters[2]),
            conv_block(filters[2], filters[2]),
        )
        self.attnlayer_ir = nn.Sequential(
            conv_block(3 * filters[2], filters[2]),
            conv_block(filters[2], 1),
            nn.Conv2d(in_channels=1, out_channels=1, stride=1, kernel_size=1, padding=0)
        )
        self.attnlayer_vi = nn.Sequential(
            conv_block(3 * filters[2], filters[2]),
            conv_block(filters[2], 1),
            nn.Conv2d(in_channels=1, out_channels=1, stride=1, kernel_size=1, padding=0)
        )
        self.decoder = nn.Sequential(
            conv_block(filters[2], filters[1]),
            conv_block(filters[1], filters[0]),
            conv_block(filters[0], out_ch)
        )
        self.Decoder_ir = ConvRelu2d(filters[2], 1, kernel_size=1, stride=1, padding=0)
        self.Decoder_vi = ConvRelu2d(filters[2], 1, kernel_size=1, stride=1, padding=0)

    def forward(self, vis, ir):
        x_ir = self.fuse_ir(torch.cat((vis, ir), dim=1))
        x_vi = self.fuse_vi(torch.cat((vis, ir), dim=1))

        x_ir_sa = x_ir * self.sa_ir(x_ir)
        x_ir_ca = x_ir * self.ca_ir(x_ir)
        attn_ir = self.sigmoid(self.attnlayer_ir(torch.cat((x_ir_sa, x_ir, x_ir_ca), dim=1)))

        x_vi_sa = x_vi * self.sa_vi(x_vi)
        x_vi_ca = x_vi * self.ca_vi(x_vi)
        attn_vi = self.sigmoid(self.attnlayer_vi(torch.cat((x_vi_sa, x_vi, x_vi_ca), dim=1)))

        input = vis * attn_vi + ir * attn_ir

        re_ir = self.Decoder_ir(input)
        re_vi = self.Decoder_vi(input)
        out = self.decoder(input)
        # out = normPRED(out)
        # re_ir = normPRED(re_ir)
        # re_vi = normPRED(re_vi)
        return out, re_ir, re_vi

class Decoder_sg(nn.Module):
    def __init__(self, n_classes=9):
        super(Decoder_sg, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.sa_ir = SpatialAttention()
        self.sa_vi = SpatialAttention()
        self.ca_ir = Ca(filters[2])
        self.ca_vi = Ca(filters[2])
        self.fuse_ir = nn.Sequential(
            conv_block(2 * filters[2], filters[2]),
            conv_block(filters[2], filters[2]),
        )
        self.fuse_vi = nn.Sequential(
            conv_block(2 * filters[2], filters[2]),
            conv_block(filters[2], filters[2]),
        )
        self.attnlayer_ir = nn.Sequential(
            conv_block(3 * filters[2], filters[2]),
            conv_block(filters[2], 1),
            nn.Conv2d(in_channels=1, out_channels=1, stride=1, kernel_size=1, padding=0)
        )
        self.attnlayer_vi = nn.Sequential(
            conv_block(3 * filters[2], filters[2]),
            conv_block(filters[2], 1),
            nn.Conv2d(in_channels=1, out_channels=1, stride=1, kernel_size=1, padding=0)
        )
        self.decoder = nn.Sequential(
            conv_block(filters[2], filters[1]),
            conv_block(filters[1], filters[0]),
            conv_block(filters[0], n_classes),
            nn.Conv2d(in_channels=n_classes, out_channels=n_classes, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, vis, ir):
        x_ir = self.fuse_ir(torch.cat((vis, ir), dim=1))
        x_vi = self.fuse_vi(torch.cat((vis, ir), dim=1))

        x_ir_sa = x_ir * self.sa_ir(x_ir)
        x_ir_ca = x_ir * self.ca_ir(x_ir)
        attn_ir = self.sigmoid(self.attnlayer_ir(torch.cat((x_ir_sa, x_ir, x_ir_ca), dim=1)))

        x_vi_sa = x_vi * self.sa_vi(x_vi)
        x_vi_ca = x_vi * self.ca_vi(x_vi)
        attn_vi = self.sigmoid(self.attnlayer_vi(torch.cat((x_vi_sa, x_vi, x_vi_ca), dim=1)))

        input = vis * attn_vi + ir * attn_ir
        out = self.decoder(input)
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