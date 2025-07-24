import torch
import torch.nn as nn
import torch.nn.functional as F


class DR(nn.Module):
    def __init__(self, in_d, out_d):
        super(DR, self).__init__()
        self.in_d = in_d
        self.out_d = out_d
        self.conv1 = nn.Conv2d(self.in_d, self.out_d, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.out_d)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        return x

class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(
            1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        # 24, 1, 1, 1
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.transpose(-1, -3)).transpose(
            -1, -3)
        # y = self.conv(y.squeeze(-1).transpose(-1, -3)).transpose(
        #     -1, -3).unsqueeze(-1)
        # Multi-scale information fusion
        y = self.sigmoid(y)

        return y.expand_as(x)

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio = 16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels,in_channels//ratio,1,bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels//ratio, in_channels,1,bias=False)
        self.sigmod = nn.Sigmoid()
    def forward(self,x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmod(out)

class Decoder(nn.Module):
    def __init__(self, fc, BatchNorm):
        super(Decoder, self).__init__()
        self.fc = fc
        self.dr1 = DR(512, 96)
        self.dr2 = DR(256, 96)
        self.dr3 = DR(128, 96)
        self.dr4 = DR(64, 96)


        self.ca = eca_layer(384)
        self.ca1 = eca_layer(96)
        self.max_pool = nn.AdaptiveMaxPool2d(128)

        self.last_conv = nn.Sequential(nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(384),
                                       nn.ReLU(),
                                       # nn.Dropout(0.2),
                                       nn.Conv2d(384, 128, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(128),
                                       nn.ReLU(),
                                       # nn.Dropout(0.2),
                                       nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(128),
                                       nn.ReLU(),
                                       # nn.Dropout(0.2),
                                       nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(32),
                                       nn.ReLU(),
                                       # nn.Dropout(0.2),
                                       )
        self.classifier = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm(32),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Conv2d(32, self.fc, kernel_size=1, stride=1, padding=0, bias=False),
            BatchNorm(self.fc),
            nn.ReLU(),
        )

        self._init_weight()

    def forward(self, x1, x2, x3, x4):
        x1 = self.dr1(x1)
        x2 = self.dr2(x2)
        x3 = self.dr3(x3)
        x4 = self.dr4(x4)

        x1 = F.interpolate(x1, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x2 = F.interpolate(x2, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x3 = F.interpolate(x3, size=x4.size()[2:], mode='bilinear', align_corners=True)

        out = torch.cat((x2, x3, x4, x1), dim=1)

        m = self.last_conv(out)
        n = self.classifier(m)
        return n

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def build_decoder(fc, BatchNorm):
    return Decoder(fc, BatchNorm)