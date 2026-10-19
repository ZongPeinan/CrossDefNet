import torch
import torch.nn as nn
import torch.nn.functional as F


class CSAF(nn.Module):

    def __init__(self, channels_list, c2):

        super().__init__()
        self.c2 = c2
        self.n = len(channels_list)

        # 1x1 conv projections to unify channel dimensions across stages
        self.projs = nn.ModuleList()
        for c in channels_list:
            if c != c2:
                self.projs.append(nn.Conv2d(c, c2, 1, bias=False))
            else:
                self.projs.append(nn.Identity())

        # Learnable pseudo-query: zero-initialized per paper §5 "Crucially, all
        # pseudo-query vectors must be initialized to zero. This ensures that the
        # initial attention weights are uniform, which reduces AttnRes to an
        # equal-weight average at the start of training."
        self.query = nn.Parameter(torch.zeros(c2))

    def forward(self, x_list):

        target = x_list[-1]
        B, _, H, W = target.shape

        # Project to common channels & align spatial resolution to target
        aligned = []
        for x, proj in zip(x_list, self.projs):
            x = proj(x)
            if x.shape[2] != H or x.shape[3] != W:
                x = F.interpolate(x, (H, W), mode='bilinear', align_corners=False)
            aligned.append(x)

        # Stack sources: (N, B, C, H, W)
        V = torch.stack(aligned, dim=0)

        # RMSNorm on keys (per paper: prevents large-magnitude outputs from dominating)
        rms = torch.sqrt(V.pow(2).mean(dim=2, keepdim=True) + 1e-6)
        K = V / rms

        # Attention: pseudo-query (C,) dot keys (N, B, C, H, W) → logits (N, B, H, W)
        logits = torch.einsum('c, n b c h w -> n b h w', self.query, K)
        alpha = logits.softmax(dim=0)  # (N, B, H, W)

        # Weighted sum over sources
        out = torch.einsum('n b h w, n b c h w -> b c h w', alpha, V)
        return out