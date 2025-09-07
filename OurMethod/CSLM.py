import torch
import torch.nn as nn

def compute_pixelwise_similarity(x, y):
    """
    计算每个像素位置上每个通道的相似度，并返回增强后的张量。

    参数:
    - x, y: 输入张量，形状为 [1, C, H, W]

    返回:
    - sim_x, sim_y: 增强相似部分的张量
    - diff_x, diff_y: 增强不同部分的张量
    """
    assert x.shape == y.shape, "Input tensors must have the same shape."

    batch_size, channels, height, width = x.shape
    x_flat = x.view(batch_size, channels, -1)  # [1, C, H*W]
    y_flat = y.view(batch_size, channels, -1)  # [1, C, H*W]

    dot_product = torch.sum(x_flat * y_flat, dim=1, keepdim=True)  # [1, 1, H*W]
    norm_x = torch.norm(x_flat, dim=1, keepdim=True)  # [1, 1, H*W]
    norm_y = torch.norm(y_flat, dim=1, keepdim=True)  # [1, 1, H*W]
    similarity = dot_product / (norm_x * norm_y + 1e-8)  # [1, 1, H*W]


    similarity = similarity.view(batch_size, 1, height, width)  # [1, 1, H, W]


    similarity = (similarity + 1) / 2  # 归一化到 [0, 1]


    similar_weight = similarity  # a
    dissimilar_weight = 1 - similarity  # 1 - a

    return similar_weight, dissimilar_weight

class DCM(nn.Module):
    def __init__(self, in_channels):
        super(DCM, self).__init__()
        self.dconv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, padding=0),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(inplace=True)
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + in_channels + in_channels, in_channels + in_channels + in_channels, 1, padding=0),
            nn.BatchNorm2d(in_channels + in_channels + in_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels + in_channels + in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(inplace=True),
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x, y):
        d = abs(x-y)
        dif1 = self.sigmoid(self.dconv(d))
        sim, dissim = compute_pixelwise_similarity(x, y)
        x = x * sim
        y = y * sim
        x = x*dif1*dissim
        y = y*dif1*dissim
        return x,y

if __name__ == "__main__":
    net = DCM(256)
    x = torch.randn(1, 256, 16, 16)
    y = torch.randn(1, 256, 16, 16)
    d = torch.randn(1, 256, 16, 16)
    x,y=net(x,y)
    print(x.size())
    print(y.size())




