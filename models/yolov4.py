import math
from torch import nn
import torch
from models.common import Conv, CSPBlock, SPPBlock
from models.yolo_layer import YOLOHead


class CSPDarknet(nn.Module):
    def __init__(self):
        super().__init__()

        # 3,608,608 -> 32,608,608
        self.conv1 = Conv(3, 32, k=3, s=1)

        # 32,608,608 -> 64,304,304
        self.CSPBlock_1 = CSPBlock(32, 64, n=1, first=True)
        # 64,304,304 -> 128,152,152
        self.CSPBlock_2 = CSPBlock(64, 128, n=2)
        # 128,152,152 -> 256,76,76
        self.CSPBlock_3 = CSPBlock(128, 256, n=8)
        # 256,76,76 -> 512,38,38
        self.CSPBlock_4 = CSPBlock(256, 512, n=8)
        # 512,38,38 -> 1024,19,19
        self.CSPBlock_5 = CSPBlock(512, 1024, n=4)

        # 初始化权值
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.CSPBlock_1(x)
        x = self.CSPBlock_2(x)
        out3 = self.CSPBlock_3(x)
        out4 = self.CSPBlock_4(out3)
        out5 = self.CSPBlock_5(out4)

        return out5, out4, out3


class FANNeck(nn.Module):
    """FAN neck"""
    def __init__(self, class_num):
        super().__init__()
        self.class_num = class_num

        # 3xCBL(131)  1024>512>1024>512
        self.conv1 = Conv(1024, 512, 1, 1, act="leaky")     # backbone out0
        self.conv2 = Conv(512, 1024, 3, 1, act="leaky")
        self.conv3 = Conv(1024, 512, 1, 1, act="leaky")

        # SSP >2048x19x19
        self.SSP = SPPBlock(5, 9, 13)

        # 3xCBL(131)  2048>512>1024>512
        self.conv4 = Conv(2048, 512, 1, 1, act="leaky")
        self.conv5 = Conv(512, 1024, 3, 1, act="leaky")
        self.conv6 = Conv(1024, 512, 1, 1, act="leaky")

        # UpSample0
        self.conv7 = Conv(512, 256, 1, 1, act="leaky")
        self.UpSample0 = nn.Upsample(scale_factor=2, mode="nearest")

        # CBL(1x1) 512x38x38 > 256x38x38
        self.conv8 = Conv(512, 256, 1, 1, act="leaky")      # backbone out1

        # 5xCBL(13131) 512>256>512>256>512>256
        self.conv9 = Conv(512, 256, 1, 1, act="leaky")
        self.conv10 = Conv(256, 512, 3, 1, act="leaky")
        self.conv11 = Conv(512, 256, 1, 1, act="leaky")
        self.conv12 = Conv(256, 512, 3, 1, act="leaky")
        self.conv13 = Conv(512, 256, 1, 1, act="leaky")

        # UpSample1
        self.conv14 = Conv(256, 128, 1, 1, act="leaky")
        self.UpSample1 = nn.Upsample(scale_factor=2, mode="nearest")

        # CBL(1x1) 256x76x76 > 128x76x76
        self.conv15 = Conv(256, 128, 1, 1, act="leaky")      # backbone out2

        # 5xCBL(13131) 256>128>256>128>256>128
        self.conv16 = Conv(256, 128, 1, 1, act="leaky")
        self.conv17 = Conv(128, 256, 3, 1, act="leaky")
        self.conv18 = Conv(256, 128, 1, 1, act="leaky")
        self.conv19 = Conv(128, 256, 3, 1, act="leaky")
        self.conv20 = Conv(256, 128, 1, 1, act="leaky")

        # CBL(3x3) >256x76x76
        self.conv21 = Conv(128, 256, 3, 1, act="leaky")
        self.last_cv2 = nn.Conv2d(256, 3*(5+self.class_num), 1, 1)  # out2

        # DownSample2
        self.DownSample2 = Conv(128, 256, 3, 2, act="leaky")

        # 5xCBL(13131) 512>256>512>256>512>256
        self.conv22 = Conv(512, 256, 1, 1, act="leaky")
        self.conv23 = Conv(256, 512, 3, 1, act="leaky")
        self.conv24 = Conv(512, 256, 1, 1, act="leaky")
        self.conv25 = Conv(256, 512, 3, 1, act="leaky")
        self.conv26 = Conv(512, 256, 1, 1, act="leaky")

        # CBL(3x3) >512x38x38
        self.conv27 = Conv(256, 512, 3, 1, act="leaky")     # out1
        self.last_cv1 = nn.Conv2d(512, 3*(5+self.class_num), 1, 1)  # out1

        # DownSample1
        self.DownSample1 = Conv(256, 512, 3, 2, act="leaky")

        # 5xCBL(13131) 1024>512>1024>512>1024>512
        self.conv28 = Conv(1024, 512, 1, 1, act="leaky")
        self.conv29 = Conv(512, 1024, 3, 1, act="leaky")
        self.conv30 = Conv(1024, 512, 1, 1, act="leaky")
        self.conv31 = Conv(512, 1024, 3, 1, act="leaky")
        self.conv32 = Conv(1024, 512, 1, 1, act="leaky")

        # CBL(3x3) >1024x19x19
        self.conv33 = Conv(512, 1024, 3, 1, act="leaky")    # out0
        self.last_cv0 = nn.Conv2d(1024, 3*(5+self.class_num), 1, 1)  # out0

    def forward(self, x0, x1, x2):

        x0 = self.conv1(x0)     # backbone out0
        x0 = self.conv2(x0)
        x0 = self.conv3(x0)
        x0 = self.SSP(x0)

        x0 = self.conv4(x0)
        x0 = self.conv5(x0)
        t0 = self.conv6(x0)     # t0

        t0_1 = self.conv7(t0)
        up0 = self.UpSample0(t0_1)

        x1 = self.conv8(x1)     # backbone out1
        x1 = torch.cat([x1, up0], dim=1)

        x1 = self.conv9(x1)
        x1 = self.conv10(x1)
        x1 = self.conv11(x1)
        x1 = self.conv12(x1)
        t1 = self.conv13(x1)    # t1

        t1_1 = self.conv14(t1)
        up1 = self.UpSample1(t1_1)

        x2 = self.conv15(x2)    # backbone out2
        x2 = torch.cat([x2, up1], dim=1)

        x2 = self.conv16(x2)
        x2 = self.conv17(x2)
        x2 = self.conv18(x2)
        x2 = self.conv19(x2)
        t2 = self.conv20(x2)    # t2

        out2 = self.conv21(t2)
        out2 = self.last_cv2(out2)  # out2

        d2 = self.DownSample2(t2)
        t1 = torch.cat([t1, d2], dim=1)

        t1 = self.conv22(t1)
        t1 = self.conv23(t1)
        t1 = self.conv24(t1)
        t1 = self.conv25(t1)
        t1 = self.conv26(t1)

        out1 = self.conv27(t1)
        out1 = self.last_cv1(out1)  # out1

        d1 = self.DownSample1(t1)
        t0 = torch.cat([t0, d1], dim=1)

        t0 = self.conv28(t0)
        t0 = self.conv29(t0)
        t0 = self.conv30(t0)
        t0 = self.conv31(t0)
        t0 = self.conv32(t0)

        out0 = self.conv33(t0)  # out0
        out0 = self.last_cv0(out0)  # out0

        return out0, out1, out2


class YOLOv4(nn.Module):
    """CSPDarknet + SPP + FAN + YOLO"""
    def __init__(self, class_num, image_size=416, anchors=None, anchor_masks=None):
        super().__init__()
        if anchors is None:
            # 默认在 image_size=608 的anchors
            if image_size == 608:
                anchors = [[12, 16], [19, 36], [40, 28],
                           [36, 75], [76, 55], [72, 146],
                           [142, 110], [192, 243], [459, 401]]
            elif image_size == 416:
                anchors = [[10, 13], [16, 30], [33, 23],
                           [30, 61], [62, 45], [59, 119],
                           [116, 90], [156, 198], [373, 326]]

        if anchor_masks is None:
            anchor_masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

        self.anchors = torch.tensor(anchors)
        self.anchor_masks = torch.tensor(anchor_masks)
        self.class_num = class_num
        self.image_size = image_size

        self.backbone = CSPDarknet()
        self.neck = FANNeck(self.class_num)
        self.yolo_layers = self._create_yolo_head(3)

        # 此处需要将yolo_layers列表展开，作为yolo4的属性，才能设置他们训练与否
        self.yolo_0 = self.yolo_layers[0]
        self.yolo_1 = self.yolo_layers[1]
        self.yolo_2 = self.yolo_layers[2]

    def forward(self, x):
        x0, x1, x2 = self.backbone(x)
        out0, out1, out2 = self.neck(x0, x1, x2)
        out0 = self.yolo_0(out0, self.image_size)
        out1 = self.yolo_1(out1, self.image_size)
        out2 = self.yolo_2(out2, self.image_size)

        if not self.training:
            return torch.cat([out0, out1, out2], dim=1)

        return [out0, out1, out2]

    def _create_yolo_head(self, n=3):
        layers = []
        for i in range(n):
            layer = YOLOHead(self.anchors[self.anchor_masks[i]], self.class_num)
            layers.append(layer)
        return layers


if __name__ == '__main__':

    x = torch.ones((1, 3, 608, 608)).to(torch.device("cuda"))

    model = YOLOv4(class_num=1, image_size=416).to(torch.device("cuda"))
    # model.eval()

    out = model(x)

    print(out[0].shape)
    print(out[1].shape)
    print(out[2].shape)







