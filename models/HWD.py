import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward


class HWD(nn.Module):
    def __init__(self, c1, c2):

        super().__init__()

        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(c1 * 4, c2, kernel_size=1, stride=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):

        x = x.float()
        yL, yH = self.wt(x)
        y_HL = yH[0][:, :, 0, ::]
        y_LH = yH[0][:, :, 1, ::]
        y_HH = yH[0][:, :, 2, ::]
        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        x = self.conv_bn_relu(x)
        return x