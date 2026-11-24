import torch.nn as nn
import torch.nn.functional as F


def cbr(i, o, k=3, s=1, g=1):
    return nn.Sequential(
        nn.Conv2d(i, o, k, s, k // 2, groups=g, bias=False),
        nn.BatchNorm2d(o),
        nn.ReLU(inplace=True))


def dsconv(i, o, s=1):
    return nn.Sequential(cbr(i, i, 3, s, g=i), cbr(i, o, 1))


class FSCN(nn.Module):
    def __init__(self, num_classes=25, in_ch=3):
        super().__init__()
        self.detail = nn.Sequential(cbr(in_ch, 32, 3, 2), dsconv(32, 48, 2), dsconv(48, 64, 2))
        self.deep = nn.Sequential(
            dsconv(64, 96, 2), dsconv(96, 96),
            dsconv(96, 128, 2), dsconv(128, 128), dsconv(128, 128))
        self.ctx = nn.Sequential(nn.AdaptiveAvgPool2d(1), cbr(128, 128, 1))
        self.proj = cbr(128, 64, 1)
        self.fuse = dsconv(64, 64)
        self.head = nn.Sequential(dsconv(64, 64), nn.Dropout2d(0.1),
                                  nn.Conv2d(64, num_classes, 1))

    def forward(self, x):
        size = x.shape[2:]
        f = self.detail(x)
        d = self.deep(f)
        d = self.proj(d + self.ctx(d))
        d = F.interpolate(d, size=f.shape[2:], mode="bilinear", align_corners=False)
        y = self.head(self.fuse(f + d))
        return F.interpolate(y, size=size, mode="bilinear", align_corners=False)