import torch
import torch.nn as nn
from .conv import Conv
import torch.nn.functional as F



class DGMR(nn.Module):
    """Dual-Gate Multi-path Residual block.

    Args:
        c1 (int): Input channels.
        c2 (int): Output channels.
        e (float): Hidden expansion ratio.
        branch_temp (float): Softmax temperature for branch weighting.
            >1.0 makes weights flatter and usually more conservative.
        gate_gain (float | None): If None, keep original [0,1] multiplicative gates.
            If set (e.g. 0.2), remap gates to mild bidirectional scaling around 1.0.
        init_delta (float): Initial residual scale.
    """

    def __init__(self, c1, c2, e=0.375, branch_temp=1.0, gate_gain=None, init_delta=0.0):
        super().__init__()
        hidden = max(16, int(c2 * e))
        self.branch_temp = max(float(branch_temp), 1e-3)
        self.gate_gain = None if gate_gain is None else float(gate_gain)

        self.stem = Conv(c1, c2, k=1, s=1, act=False) if c1 != c2 else nn.Identity()
        self.reduce = Conv(c2, hidden, k=1, s=1)

        self.path_local = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
        )
        self.path_context = nn.Sequential(
            nn.Conv2d(hidden, hidden, 5, 1, 2, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
        )
        self.path_proj = Conv(hidden * 3, c2, k=1, s=1, act=False)

        # Dynamic branch weighting.
        self.branch_gate = nn.Conv2d(hidden * 3, 3, 1, 1, 0)

        squeeze = max(c2 // 8, 16)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c2, squeeze, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(squeeze, c2, 1, bias=True),
            nn.Sigmoid(),
        )

        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, 7, 1, 3, bias=False),
            nn.Sigmoid(),
        )

        self.delta = nn.Parameter(torch.full((1,), float(init_delta)))
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        base = self.stem(x)
        h = self.reduce(base)

        h = torch.nan_to_num(h)
        local = self.path_local(h)
        context = self.path_context(h)
        contrast = h - F.avg_pool2d(h, kernel_size=3, stride=1, padding=1)
        contrast = torch.nan_to_num(contrast)

        merged = torch.cat((local, context, contrast), dim=1)
        logits = self.branch_gate(torch.nan_to_num(merged))
        alpha = torch.softmax(logits / self.branch_temp, dim=1)
        alpha = torch.nan_to_num(alpha)
        fused = alpha[:, 0:1] * local + alpha[:, 1:2] * context + alpha[:, 2:3] * contrast
        out = self.path_proj(torch.cat((local, context, fused), dim=1))
        out = torch.nan_to_num(out)

        cg = self.channel_gate(out)
        sg = self.spatial_gate(torch.cat((out.mean(dim=1, keepdim=True), out.max(dim=1, keepdim=True)[0]), dim=1))
        if self.gate_gain is None:
            out = out * cg * sg
        else:
            gain = self.gate_gain
            cg = 1.0 + gain * (cg - 0.5) * 2.0
            sg = 1.0 + gain * (sg - 0.5) * 2.0
            out = out * torch.clamp(cg * sg, min=0.1, max=2.0)

        return self.act(base + self.delta * out)