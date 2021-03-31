import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftmaxRankingLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets):
        # input check
        assert inputs.shape == targets.shape

        # compute the probabilities
        probs = F.softmax(inputs + 1e-8, dim=0)
        # reduction
        loss = -torch.sum(torch.log(probs + 1e-8) * targets, dim=0).mean()

        return loss


class RankingLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(RankingLoss, self).__init__()
        self.m = 0.2
        self.gamma = 64
        self.reduction = reduction
        self.soft_plus = nn.Softplus()

    def forward(self, sim, label):
        loss_v = 0
        loss_l = 0
        loss_loc = 0
        batch_size = label.shape[0]
        delta_p = 1 - self.m
        delta_n = self.m

        # torch.set_printoptions(precision=5, sci_mode=False)
        for i in range(batch_size):
            temp_label = label[i]
            index = temp_label > 0.5
            index = index.nonzero().squeeze(1)
            if index.shape[0] > 0:
                pos_sim = torch.index_select(sim[i], 0, index)
                alpha_p = torch.clamp(0.8 - pos_sim.detach(), min=0)
                logit_p = - alpha_p * (pos_sim - delta_p) * self.gamma
            else:
                logit_p = torch.zeros(1)[0].cuda()

            # pos_sim = torch.index_select(loc_sim[i], 0, index)
            # alpha_p = torch.clamp(1 + self.m - pos_sim.detach(), min=0)
            # logit_p_loc = - alpha_p * (pos_sim - delta_p) * self.gamma

            # neg_l_sim = torch.index_select(neg_sim[i], 0, index)
            # alpha_n = torch.clamp(neg_l_sim.detach() + self.m, min=0)
            # logit_n = alpha_n * (neg_l_sim - delta_n) * self.gamma
            # loss_l += self.soft_plus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))

            index = (temp_label < 0.25) # * objectness_masks[0]
            index = (index).nonzero().squeeze(1)
            # idx = torch.randint(0, index.shape[0], (10,))
            # index = index[idx]

            neg_v_sim = torch.index_select(sim[i], 0, index)
            if neg_v_sim.shape[0] > 20:
                index = neg_v_sim.topk(10, largest=True)[1]
                neg_v_sim = torch.index_select(neg_v_sim, 0, index)

            alpha_n = torch.clamp(neg_v_sim.detach() - 0.2, min=0)
            logit_n = alpha_n * (neg_v_sim - delta_n) * self.gamma
            # print('logit_n', logit_n)
            # print('loss_n', torch.logsumexp(logit_n, dim=0))
            # print('loss_p', torch.logsumexp(logit_p, dim=0))
            # print('loss_v', torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))

            # neg_loc_sim = torch.index_select(loc_sim[i], 0, index)
            # index = neg_loc_sim.topk(20, largest=True)[1]
            # neg_loc_sim = torch.index_select(neg_loc_sim, 0, index)
            # alpha_n = torch.clamp(neg_loc_sim.detach() + self.m, min=0)
            # logit_n = alpha_n * (neg_loc_sim - delta_n) * self.gamma

            loss_loc += self.soft_plus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))

        if self.reduction == 'mean':
            loss = (loss_l + loss_v + loss_loc) / batch_size
        return loss


class SimCLRLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(SimCLRLoss, self).__init__()
        self.m = 0.2
        self.gamma = 64
        self.reduction = reduction
        self.soft_plus = nn.Softplus()

    def forward(self, sim, label):
        sim = torch.exp(7 * sim)
        loss = - torch.log((sim * label).sum() / (sim.sum() - (sim * label).sum() + 1e-8))

        return loss


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.2, gamma=5, reduction='mean'):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.gamma = gamma
        self.reduction = reduction
        self.soft_plus = nn.Softplus()

    def forward(self, score, label):
        score *= self.gamma
        sim = (score*label).sum()
        neg_sim = score*label.logical_not()
        neg_sim = torch.logsumexp(neg_sim, dim=0) # soft max
        loss = torch.clamp(neg_sim - sim + self.margin, min=0).sum()
        return loss


class SegLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, weight=None, ignore_index=255):
        super(SegLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.bce_fn = nn.BCEWithLogitsLoss(weight=self.weight)

    def forward(self, preds, labels):
        if self.ignore_index is not None:
            mask = labels != self.ignore_index
            labels = labels[mask]
            preds = preds[mask]

        logpt = -self.bce_fn(preds, labels)
        pt = torch.exp(logpt)
        loss = -((1 - pt)**self.gamma) * self.alpha * logpt
        return loss