# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.data.exp()

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class LabelSmoothing(nn.Module):
    # Implement label smoothing.  size表示类别总数
    def __init__(self, size, smoothing=0.0, reduction='batchmean'):
        super(LabelSmoothing, self).__init__()

        self.criterion = nn.KLDivLoss(size_average=False, reduction=reduction)

        # self.padding_idx = padding_idx

        self.confidence = 1.0 - smoothing

        self.smoothing = smoothing

        self.size = size

        self.true_dist = None

    def forward(self, x, target):
        """
        x表示输入 (N，M)N个样本，M表示总类数，每一个类的概率log P
        target表示label（M，）
        """
        assert x.size(1) == self.size
        true_dist = x.data.clone()  # 先深复制过来
        # print(true_dist)
        true_dist.fill_(self.smoothing / (self.size - 1))  # otherwise的公式
        # print(true_dist)

        # 变成one-hot编码，1表示按列填充，
        # target.data.unsqueeze(1)表示索引,confidence表示填充的数字
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)

        true_dist.requires_grad = False
        self.true_dist = true_dist

        return self.criterion(x, true_dist)


class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.

    Reference:
        Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

    Imported from `<https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py>`_.

    Args:
        margin (float, optional): margin for triplet. Default is 0.3.
    """

    def __init__(self, margin=0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        if margin != 0:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): feature matrix with shape (batch_size, feat_dim).
            targets (torch.LongTensor): ground truth labels with shape (batch_szie).
        """
        n = inputs.size(0)
        # Compute pairwise distance, replace by the official when merged
        inputs = self.normalize(inputs)
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())      # x^2+y^2-2xy
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t()) # if an element==1, it is a ap;
        dist_ap, dist_an = [], []

        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        count = self.count_valid(dist, mask, n)
        if self.margin != 0:
            return self.ranking_loss(dist_an, dist_ap, y), count
        else:
            return self.ranking_loss(dist_an - dist_ap, y), count

    @staticmethod
    def normalize(x, axis=1):
        x = 1.0 * x / (torch.norm(x, 2, axis, keepdim=True) + 1e-12)
        return x

