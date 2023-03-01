import torch
from torch import nn

from models.yolo_layer import YOLOHead
from models.yolov3_spp import FPNNeck
from models.yolov4 import CSPDarknet


class YOLOv4FPN(nn.Module):
    """CSPDarknet + SPP + FPN + YOLO"""
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
    x = torch.ones((1, 3, 608, 608)).to(torch.device("cuda"))

    model = YOLOv4FPN(class_num=1, image_size=416).to(torch.device("cuda"))

    # model.eval()

    out = model(x)

    # print(out.shape)

    print(out[0].shape)
    print(out[1].shape)
    print(out[2].shape)
















