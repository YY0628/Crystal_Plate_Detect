import math
from collections import OrderedDict

import torch
from torch import nn
from models.common import Conv, ResBlock, SPPBlock
from models.yolo_layer import YOLOHead


def _make_layer(in_c, out_c, k, s, blocks):
    layers = [("Conv", Conv(in_c, out_c, k, s, act="leaky"))]
    for i in range(blocks):
        layers.append(("res{}".format(i), ResBlock(out_c, out_c, act="leaky")))
    return nn.Sequential(OrderedDict(layers))


class Darknet53(nn.Module):
    """
    darknet53 backbone
    """
    def __init__(self):
        super().__init__()

        self.layer0 = nn.Sequential(OrderedDict([("Conv", Conv(3, 32, 3, 1, act="leaky"))]))
        self.layer1 = _make_layer(32, 64, 3, 2, blocks=1)
        self.layer2 = _make_layer(64, 128, 3, 2, blocks=2)
        self.layer3 = _make_layer(128, 256, 3, 2, blocks=8)
        self.layer4 = _make_layer(256, 512, 3, 2, blocks=8)
        self.layer5 = _make_layer(512, 1024, 3, 2, blocks=4)

        # 进行权值初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.layer0(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out3 = self.layer3(out)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)

        return out5, out4, out3


class FPNNeck(nn.Module):
    """FPN neck"""
    def __init__(self, class_num):
        super().__init__()
        self.class_num = class_num

        self.conv1 = Conv(1024, 512, 1, 1, act="leaky")
        self.conv2 = Conv(512, 1024, 3, 1, act="leaky")
        self.conv3 = Conv(1024, 512, 1, 1, act="leaky")

        self.SPP = SPPBlock(5, 9, 13)

        self.conv4 = Conv(2048, 512, 1, 1, act="leaky")
        self.conv5 = Conv(512, 1024, 3, 1, act="leaky")
        self.conv6 = Conv(1024, 512, 1, 1, act="leaky")     # t0

        self.conv7 = Conv(512, 256, 1, 1, act="leaky")
        self.UpSample0 = nn.Upsample(scale_factor=2, mode="nearest")    # up0

        self.conv8 = Conv(768, 256, 1, 1, act="leaky")
        self.conv9 = Conv(256, 512, 3, 1, act="leaky")
        self.conv10 = Conv(512, 256, 1, 1, act="leaky")
        self.conv11 = Conv(256, 512, 3, 1, act="leaky")
        self.conv12 = Conv(512, 256, 1, 1, act="leaky")     # t1

        self.conv13 = Conv(256, 128, 1, 1, act="leaky")
        self.UpSample1 = nn.Upsample(scale_factor=2, mode="nearest")    # up1

        self.conv14 = Conv(384, 128, 1, 1, act="leaky")
        self.conv15 = Conv(128, 256, 3, 1, act="leaky")
        self.conv16 = Conv(256, 128, 1, 1, act="leaky")
        self.conv17 = Conv(128, 256, 3, 1, act="leaky")
        self.conv18 = Conv(256, 128, 1, 1, act="leaky")     # t2

        self.conv19 = Conv(128, 256, 3, 1, act="leaky")
        '''最后一层千万要用普通的卷积层'''
        self.last_cv2 = nn.Conv2d(256, 3*(5+self.class_num), 1, 1)  # out2

        self.conv20 = Conv(256, 512, 3, 1, act="leaky")
        self.last_cv1 = nn.Conv2d(512, 3*(5+self.class_num), 1, 1)   # out1

        self.conv21 = Conv(512, 1024, 3, 1, act="leaky")
        self.last_cv0 = nn.Conv2d(1024, 3*(5+self.class_num), 1, 1)  # out0

    def forward(self, x0, x1, x2):
        t0 = self.conv1(x0)
        t0 = self.conv2(t0)
        t0 = self.conv3(t0)
        t0 = self.SPP(t0)
        t0 = self.conv4(t0)
        t0 = self.conv5(t0)
        t0 = self.conv6(t0)

        out0 = self.conv21(t0)
        out0 = self.last_cv0(out0)  # out0

        up0 = self.conv7(t0)
        up0 = self.UpSample0(up0)
        x1 = torch.cat([x1, up0], dim=1)

        t1 = self.conv8(x1)
        t1 = self.conv9(t1)
        t1 = self.conv10(t1)
        t1 = self.conv11(t1)
        t1 = self.conv12(t1)

        out1 = self.conv20(t1)
        out1 = self.last_cv1(out1)  # out1

        up1 = self.conv13(t1)
        up1 = self.UpSample1(up1)
        x2 = torch.cat([x2, up1], dim=1)

        t2 = self.conv14(x2)
        t2 = self.conv15(t2)
        t2 = self.conv16(t2)
        t2 = self.conv17(t2)
        t2 = self.conv18(t2)

        out2 = self.conv19(t2)
        out2 = self.last_cv2(out2)

        return out0, out1, out2


class YOLOv3SPP(nn.Module):
    """Darknet + SPP + FPN + YOLO"""
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

        self.backbone = Darknet53()
        self.neck = FPNNeck(self.class_num)
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
    device = torch.device("cuda")
    x = torch.ones((1, 3, 416, 416)).to(device)
    model = YOLOv3SPP(class_num=1, image_size=416).to(device)
    # model.eval()
    out = model(x)

    print(out[0].shape)
    print(out[1].shape)
    print(out[2].shape)

    # print(out.shape)




