import math
import torch
from torch import nn
import numpy as np
from losses.lovasz_losses import lovasz_softmax


class OHMECE(object):
    def __init__(self, thresh=0.7, weights=None, ignore_idx=255, ohme=True):
        self.loss_thresh = torch.tensor(-math.log(thresh), requires_grad=False)
        self.ce = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=ignore_idx)
        self.weights = None
        self.ohme = ohme
        if weights is not None:
            self.weights = torch.tensor(weights)

    def __call__(self, logits, labels):
        logits = logits.float()
        labels = labels.long()
        if self.loss_thresh.device != logits.device:
            self.loss_thresh.to(logits.device)
        if self.weights is not None and self.weights.device != logits.device:
            self.weights.to(logits.device)
            self.ce.weights = self.weights
        #c = logits.size(1)
        #n_min = labels.numel() // 16
        n_min = (labels[labels>0]).numel()//20 
        
        b,c,h,w = logits.shape
        #loss = self.ce(logits.permute(0, 2, 3, 1).reshape(-1, c), labels.view(-1))
        loss = self.ce(logits.permute(0, 2, 3, 1).contiguous().view(b,c,-1), labels.squeeze().contiguous().reshape(b,-1))
        if not self.ohme:
            return loss.mean()
        loss_hard = loss[loss > self.loss_thresh]
        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)
        return loss_hard.mean()


class DICE(object):
    def __init__(self):
        super(DICE, self).__init__()

    def __call__(self, logits, label):
        """
        :param logits: b c h m
        :param label: b h m
        :return:
        """
        predicts = logits.softmax(dim=1)
        label_val = torch.zeros_like(predicts)
        label_val = torch.scatter(label_val, 1, label[:, None, ...], 1)
        iou_refine = (2 * (predicts * label_val).sum(dim=(-1, -2)) + 1) / \
                     (predicts.sum(dim=(-1, -2)) + label_val.sum(dim=(-1, -2)) + 1)
        iou_loss = 1 - iou_refine
        return iou_loss.mean()


class BCE(object):
    def __init__(self):
        self.bce = torch.nn.BCELoss()

    def __call__(self, logits, label):
        """
        :param logits: n c h w
        :param label: n h w
        :return:
        """
        predicts = logits.sigmoid().squeeze(1)
        label = (label > 0).float()
        return self.bce(predicts, label)


class FocalLoss(object):
    def __init__(self, alpha=0.25, gamma=2):
        self.alpha = alpha
        self.gamma = gamma

    def __call__(self, logits, label):
        b, c, h, w = logits.shape
        predicts = logits.sigmoid()
        targets = torch.zeros(size=(b, c + 1, h, w), device=logits.device)
        targets = torch.scatter(targets, 1, label[:, None, ...], 1.0)[:, 1:, ...]
        positive_loss = -targets * self.alpha * ((1 - predicts) ** self.gamma) * predicts.clip(min=1e-5).log()
        negative_loss = -(1 - targets) * (1 - self.alpha) * (predicts ** self.gamma) * (1 - predicts).clip(
            min=1e-5).log()
        f_loss = (positive_loss + negative_loss).sum() / (targets.sum() + 1)
        return f_loss


class MultiDICE(object):
    def __init__(self):
        super(MultiDICE, self).__init__()

    def __call__(self, logits, label):
        b, c, h, w = logits.shape
        targets = torch.zeros(size=(b, c + 1, h, w), device=logits.device)
        targets = torch.scatter(targets, 1, label[:, None, ...], 1.0)[:, 1:, ...]
        predicts = logits.sigmoid()
        dice = ((2 * predicts * targets).sum(dim=(-1, -2)) + 1) / \
               (predicts.sum(dim=(-1, -2)) + targets.sum(dim=(-1, -2)) + 1)
        dice_loss = 1 - dice
        return dice_loss.mean()


class MultiBCE(object):
    def __init__(self):
        self.bce = torch.nn.BCELoss()

    def __call__(self, logits, label):
        b, c, h, w = logits.shape
        targets = torch.zeros(size=(b, c + 1, h, w), device=logits.device)
        targets = torch.scatter(targets, 1, label[:, None, ...], 1.0)[:, 1:, ...]
        predicts = logits.sigmoid()
        loss = self.bce(predicts, targets)
        return loss


class LovaszSoftMax(object):
    def __init__(self, ignore_idx=255):
        self.ignore_idx = ignore_idx

    def __call__(self, logits, mask):
        probs = logits.softmax(dim=1)
        loss = lovasz_softmax(probs, mask, ignore=self.ignore_idx)
        return loss


class BDice(object):
    def __init__(self):
        super(BDice, self).__init__()

    def __call__(self, predicts, targets):
        """
        :param predicts: no logits [b,c,h,w]
        :param targets:
        :return:
        """
        logits = predicts.sigmoid()
        dice = (2 * logits * targets + 1e-5).sum((-1, -2, -3)) / \
               ((logits ** 2).sum((-1, -2, -3)) + (targets ** 2).sum((-1, -2, -3)) + 1e-5)
        return (1 - dice).mean()


class DetailLoss(object):
    def __init__(self, thresh=0.128):
        self.thresh = thresh
        self.bce = torch.nn.BCEWithLogitsLoss()
        self.dice = BDice()

    def __call__(self, predicts, targets):
        b, _, _, _ = predicts.shape
        targets = ((targets - self.thresh) * 64).sigmoid()
        bce_loss = self.bce(predicts.view(b, -1), targets.view(b, -1))
        dice_loss = self.dice(predicts, targets)
        return bce_loss + dice_loss


class ChannelWiseKL(nn.Module):
    def __init__(self, t=3, alpha=3):
        super(ChannelWiseKL, self).__init__()
        self.t = t
        self.alpha = alpha
        self.kl = nn.KLDivLoss()

    def forward(self, predict, target):
        """
        :param predict: [N,C,H,W]
        :param target: [N,C,H,W]
        :return:
        """
        n, c, h, w = predict.shape
        predict_prob = (predict.view(n, c, -1) / self.t).log_softmax(dim=-1)
        target_prob = (target.view(n, c, -1) / self.t).softmax(dim=-1)
        kl_loss = self.alpha * self.kl(predict_prob, target_prob)
        return kl_loss


class PixelWiseKL(nn.Module):
    def __init__(self, t=1, alpha=0.5):
        super(PixelWiseKL, self).__init__()
        self.t = t
        self.alpha = alpha
        self.kl = nn.KLDivLoss()

    def forward(self, predict, target):
        n, c, h, w = predict.shape
        predict_prob_log = (predict / self.t).log_softmax(dim=1).permute(0, 2, 3, 1).contiguous().view(-1, c)
        target_prob = (target / self.t).softmax(dim=1).permute(0, 2, 3, 1).contiguous().view(-1, c)
        kl_loss = self.alpha * self.kl(predict_prob_log, target_prob)
        return kl_loss


class PairWiseL2(nn.Module):
    def __init__(self, alpha=0.1, kernel=4, stride=4):
        super(PairWiseL2, self).__init__()
        self.alpha = alpha
        self.pool = nn.MaxPool2d(kernel_size=kernel, stride=stride, padding=0, ceil_mode=True)

    def forward(self, predict, target):
        predict_pool = self.pool(predict)
        target_pool = self.pool(target)
        loss = self.alpha * self.sim_dis(predict_pool, target_pool)
        return loss

    @staticmethod
    def l2(feat):
        l2_norm = ((feat ** 2).sum(dim=1, keepdim=True) ** 0.5) + 1e-8
        return l2_norm

    @staticmethod
    def self_cosine_sim(feat):
        n, c, h, w = feat.shape
        l2_norm = PairWiseL2.l2(feat).detach()
        feat = feat / l2_norm
        feat = feat.view(n, c, -1)
        cosine = torch.einsum("icm,icn->imn", feat, feat)
        return cosine

    @staticmethod
    def sim_dis(fs, ft):
        n, c, h, w = fs.shape
        sim_dis = ((PairWiseL2.self_cosine_sim(fs) - PairWiseL2.self_cosine_sim(ft)) ** 2) / ((h * w) ** 2) / n
        return sim_dis.sum()


if __name__ == '__main__':
    inp = torch.randn(size=(2, 1, 64, 64))
    out = torch.randn(size=(2, 1, 64, 64))
    loss = DetailLoss()
    print(loss(inp, out))
