from torch import nn
import torch
from torch.nn.functional import softplus


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * (torch.tanh(softplus(x)))
        return x


def compute_padding(k, p):
    if p is None:
        p = k // 2
    return p


class Conv(nn.Module):
    # 卷积层
    def __init__(self, in_c, out_c, k, s, act="mish", p=None, g=1, eps=1e-3, momentum=0.03, inplace=True):
        super(Conv, self).__init__()
        p = compute_padding(k, p)
        self.conv = nn.Conv2d(in_c, out_c, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(out_c, eps=eps, momentum=momentum)
        if act == "mish":
            self.act = Mish()
        elif act == "relu":
            self.act = nn.ReLU(inplace=True)
        elif act == "leaky":
            self.act = nn.LeakyReLU(0.1, inplace=inplace)
        elif act is None:
            self.act = nn.Identity()
        else:
            print("activate layer wrong!!!")

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ResBlock(nn.Module):
    # 残差模块
    def __init__(self, in_c, out_c, g=1, e=0.5, act="mish"):
        super().__init__()
        c = int(out_c*e)
        self.cv1 = Conv(in_c, c, 1, 1, act)
        self.cv2 = Conv(c, out_c, 3, 1, act, g=g)

    def forward(self, x):
        return x + self.cv2(self.cv1(x))


class CSPBlock(nn.Module):
    # CSP模块
    def __init__(self, in_c, out_c, n, first=False):
        super().__init__()

        # 初始卷积，调整通道数及特征图大小
        self.downsSample = Conv(in_c, out_c, k=3, s=2)

        if first is False:
            # 快捷分支
            self.banch0_cv0 = Conv(out_c, out_c//2, k=1, s=1)

            # 残差分支
            self.banch1_cv0 = Conv(out_c, out_c//2, k=1, s=1)
            self.resBlocks = nn.Sequential(*(ResBlock(out_c//2, out_c//2, e=1.0) for _ in range(n)))
            self.banch1_cv1 = Conv(out_c//2, out_c//2, k=1, s=1)

            # 连接后的融合
            self.concat_conv = Conv(out_c, out_c, k=1, s=1)

        else:
            self.banch0_cv0 = Conv(out_c, out_c, k=1, s=1)

            self.banch1_cv0 = Conv(out_c, out_c, k=1, s=1)
            self.resBlocks = ResBlock(out_c, out_c, e=0.5)
            self.banch1_cv1 = Conv(out_c, out_c, k=1, s=1)

            self.concat_conv = Conv(out_c*2, out_c, k=1, s=1)

    def forward(self, x):
        x = self.downsSample(x)

        x0 = self.banch0_cv0(x)

        x1 = self.banch1_cv0(x)
        x1 = self.resBlocks(x1)
        x1 = self.banch1_cv1(x1)

        x = torch.cat([x1, x0], dim=1)

        x = self.concat_conv(x)

        return x


class SPPBlock(nn.Module):

    def __init__(self, k1, k2, k3):
        super().__init__()
        self.maxpool1 = nn.MaxPool2d(kernel_size=k1, stride=1, padding=k1//2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=k2, stride=1, padding=k2//2)
        self.maxpool3 = nn.MaxPool2d(kernel_size=k3, stride=1, padding=k3//2)

    def forward(self, x):
        out1 = self.maxpool1(x)
        out2 = self.maxpool2(x)
        out3 = self.maxpool3(x)
        return torch.cat([out3, out2, out1, x], dim=1)





