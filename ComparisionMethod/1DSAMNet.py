import torch
import torch.nn as nn
import torch.nn.functional as F
from cdcommodel.backbone import build_backbone
from cdcommodel.decoder import build_decoder
from cdcommodel.dsamutils import CBAM, DS_layer
#from torchsummary import summary

class DSAMNet(nn.Module):
    def __init__(self, n_class=2,  ratio = 8, kernel = 7, backbone='resnet18', output_stride=16, f_c=64, freeze_bn=False, in_c=3):
        super(DSAMNet, self).__init__()
        BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm, in_c)
        self.decoder = build_decoder(f_c, BatchNorm)

        self.cbam0 = CBAM(f_c, ratio, kernel)
        self.cbam1 = CBAM(f_c, ratio, kernel)

        self.ds_lyr2 = DS_layer(64, 32, 2, 1, n_class)
        self.ds_lyr3 = DS_layer(128, 32, 4, 3, n_class)

        if freeze_bn:
            self.freeze_bn()

    def forward(self, input1, input2):
        x_1, f2_1, f3_1, f4_1 = self.backbone(input1)
        x_2, f2_2, f3_2, f4_2 = self.backbone(input2)

        x1 = self.decoder(x_1, f2_1, f3_1, f4_1)
        x2 = self.decoder(x_2, f2_2, f3_2, f4_2)

        x1 = self.cbam0(x1)
        x2 = self.cbam1(x2) # channel = 64

        dist = F.pairwise_distance(x1, x2, keepdim=True) # channel = 1
        dist = F.interpolate(dist, size=input1.shape[2:], mode='bilinear', align_corners=True)

        ds2 = self.ds_lyr2(torch.abs(f2_1 - f2_2))
        ds3 = self.ds_lyr3(torch.abs(f3_1 - f3_2))

        return dist, ds2, ds3


    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
if __name__ == "__main__":
    #1024 256
    net = DSAMNet()#.to('cuda')
    x = torch.randn(1, 3, 1024, 1024)
    y = torch.randn(1, 3, 1024, 1024)
    a,b,c=net(x,y)
    print(a.size())

    # from thop import profile
    # flops, params = profile(net, inputs=(x, y))
    # print("FLOPs=" + str(flops / 1000 ** 3) + 'G')
    # print(params)
    # a,b,c=net(x,y)
    # print(a.size())
    # #summary(net,input_size=[(3,256,256),(3,256,256)])