import torch
from torch import nn
import torch.nn.functional as F
from cdmodel.encoderRes import build_backbone  as encoder
from cdmodel.deco import build_decoder as decoder
from cdmodel.conv import DepthwiseSeparableConvolution as dwconv
import warnings
# Suppress specific UserWarnings related to ONNX
warnings.filterwarnings("ignore", category=UserWarning, message=".*Constant folding.*")
from cdmodel.ex_bi_mamba2 import BiMamba2_2D
from cdmodel.CSLM import DCM
#double conv
class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            #torch.nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1),
            dwconv(in_ch, in_ch, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(in_ch),
            torch.nn.ReLU(inplace=True),
            #torch.nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0),
            dwconv(in_ch, out_ch, kernel_size=1, stride=1, padding=0),
            torch.nn.BatchNorm2d(out_ch),
            torch.nn.ReLU(inplace=True),
        )
    def forward(self, x):
        x = self.conv(x)
        return x
# 融合
class decat(nn.Module):
    def __init__(self, in_ch, out_ch, upsample=False):
        super(decat, self).__init__()
        self.do_upsample = upsample
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv2d = double_conv(in_ch, out_ch)

    def forward(self, x, y):
        if self.do_upsample:
            x = self.upsample(x)
        x = torch.cat((x, y), 1)
        x = self.conv2d(x)
        return x

class MBAM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MBAM, self).__init__()
        self.linear1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.dw = dwconv(out_channels,out_channels)
        self.ssm = BiMamba2_2D(out_channels, out_channels, 32)
        self.conv4 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()
        self.ln = torch.nn.BatchNorm2d(out_channels)
    def forward(self, x):
        output1 = self.linear1(x)
        output1 = self.dw(output1)
        output1 = F.silu(output1)
        output1 = self.ssm(output1)
        output1 = self.ln(output1)
        output1 = self.sigmoid(self.conv4(output1))
        return output1*x + x

class UNet(nn.Module):
    def __init__(self, ):
        super(UNet, self).__init__()
        #resnet34  resnet18
        self.encoder = encoder('resnet34', 32, nn.BatchNorm2d, 3)
        self.decoder = decoder(1, nn.BatchNorm2d)

        self.MAlayer1 = MBAM(64, 64)
        self.MAlayer2 = MBAM(128, 128)
        self.MAlayer3 = MBAM(256, 256)
        self.MAlayer4 = MBAM(512, 512)

        self.DCM1 = DCM(64)
        self.DCM2 = DCM(128)
        self.DCM3 = DCM(256)
        self.DCM4 = DCM(512)

        self.up1 = decat(192,64,upsample=True)
        self.up2 = decat(384,128,upsample=True)
        self.up3 = decat(768,256,upsample=True)

        self.f1 = decat(128,64,upsample=False)
        self.f2 = decat(256,128,upsample=False)
        self.f3 = decat(512,256,upsample=False)
        self.f4 = decat(1024,512,upsample=False)

    def forward(self, imgx, imgy):
        x4,x1,x2,x3 = self.encoder(imgx)
        y4,y1,y2,y3 = self.encoder(imgy)

        x1 = self.MAlayer1(x1)
        y1 = self.MAlayer1(y1)

        x2 = self.MAlayer2(x2)
        y2 = self.MAlayer2(y2)

        x3 = self.MAlayer3(x3)
        y3 = self.MAlayer3(y3)

        x4 = self.MAlayer4(x4)
        y4 = self.MAlayer4(y4)

        x4,y4 = self.DCM4(x4,y4)
        out1 = self.f4(x4,y4)

        x33 = self.up3(x4,x3)
        y33 = self.up3(y4,y3)
        x33,y33 = self.DCM3(x33,y33)
        out2 = self.f3(x33,y33)

        x22 = self.up2(x33, x2)
        y22 = self.up2(y33, y2)
        x22,y22 = self.DCM2(x22,y22)
        out3 = self.f2(x22,y22)

        x11 = self.up1(x22,x1)
        y11 = self.up1(y22, y1)
        x11,y11 = self.DCM1(x11,y11)
        out4 = self.f1(x11,y11)

        out = self.decoder(out1,out2,out3,out4)
        out = F.interpolate(out, size=imgx.shape[2:], mode='bicubic', align_corners=True)
        return out

if __name__ == "__main__":
    net = UNet()#.to('cuda')#USE IT WHEN SEE SUMMARY
    x = torch.randn(1, 3, 256, 256)
    y = torch.randn(1, 3, 256, 256)
    z1 =net(x,y)
    print(z1.shape)

    # from thop import profile
    # from thop import clever_format
    # flops, params = profile(net, inputs=(x, y))
    # print("FLOPs=" + str(flops / 1000 ** 3) + 'G')
    # print("FLOPs=" + str(params / 1000 ** 2) + 'M')
