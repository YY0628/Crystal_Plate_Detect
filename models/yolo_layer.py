import torch
from itertools import chain
from torch import nn


class YOLOHead(nn.Module):
    """Detection layer
    将NECK层输出的[[batch_size, 3*(5+class_num), 13, 13],
                 [batch_size, 3*(5+class_num), 26, 26],
                 [batch_size, 3*(5+class_num), 52, 52]]
    转换为 :
        training: [[batch_size, 3, 13, 13, 5+class_num],
                   [batch_size, 3, 26, 26, 5+class_num],
                   [batch_size, 3, 52, 52, 5+class_num]]
        eval:   shape: (batch_size, 3*(13*13+26*26+52*52), 5+class_num)
    """

    def __init__(self, anchors, num_classes):
        super().__init__()
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.no = num_classes + 5  # number of outputs per anchor
        self.grid = torch.zeros(1)  # TODO

        anchors = torch.tensor(list(chain(*anchors))).float().view(-1, 2)
        self.register_buffer('anchors', anchors)
        self.register_buffer('anchor_grid', anchors.clone().view(1, -1, 1, 1, 2))
        self.stride = None

    def forward(self, x, img_size):
        stride = img_size // x.size(2)
        self.stride = stride
        bs, _, ny, nx = x.shape  # x(bs,255,19,19) to x(bs,3,19,19,85)
        x = x.view(bs, self.num_anchors, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

        if not self.training:  # inference
            if self.grid.shape[2:4] != x.shape[2:4]:
                self.grid = self._make_grid(nx, ny).to(x.device)

            x[..., 0:2] = (x[..., 0:2].sigmoid() + self.grid) * stride  # xy
            x[..., 2:4] = torch.exp(x[..., 2:4]) * self.anchor_grid.to(x.device)  # wh
            x[..., 4:] = x[..., 4:].sigmoid()
            x = x.view(bs, -1, self.no)     # (bs,3,13,13,85) => (bs,3*13*13,85)

        return x

    @staticmethod
    def _make_grid(nx=13, ny=13):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

