import os
import sys

import torch
import torch.nn as nn
import numpy as np

sys.path.append(os.path.join(os.getcwd(), "lib"))  # HACK add the lib folder
from utils.box_util import get_3d_box_batch, box3d_iou_batch

FAR_THRESHOLD = 0.6
NEAR_THRESHOLD = 0.3
GT_VOTE_FACTOR = 3  # number of GT votes per point
OBJECTNESS_CLS_WEIGHTS = [0.2, 0.8]  # put larger weights on positive objectness


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

            index = (temp_label < 0.25)
            index = (index).nonzero().squeeze(1)

            neg_v_sim = torch.index_select(sim[i], 0, index)
            if neg_v_sim.shape[0] > 20:
                index = neg_v_sim.topk(10, largest=True)[1]
                neg_v_sim = torch.index_select(neg_v_sim, 0, index)

            alpha_n = torch.clamp(neg_v_sim.detach() - 0.2, min=0)
            logit_n = alpha_n * (neg_v_sim - delta_n) * self.gamma

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


def compute_scene_mask_loss(data_dict):
    pred = data_dict['seg_scores']
    ref_center_label = data_dict["ref_center_label"].cuda()

    point_min = data_dict['point_min']
    point_max = data_dict['point_max']
    batch_size = point_min.shape[0]
    ones = torch.ones(batch_size, dtype=torch.long, device='cuda')

    # label for 9 area
    first_point = point_min + (point_max - point_min) / 3
    second_point = point_min + (point_max - point_min) / 3 * 2
    result_first = torch.le(ref_center_label, first_point)
    result_second = torch.le(ref_center_label, second_point)

    label = torch.where(result_first[:, 0] & result_first[:, 1], ones * 0, ones*4)
    label = torch.where(result_first[:, 0].logical_not() & result_second[:, 0] & result_first[:, 1], ones, label)
    label = torch.where(result_second[:, 0].logical_not() & result_first[:, 1], ones*2, label)
    label = torch.where(result_first[:, 0] & result_first[:, 1].logical_not() & result_second[:, 0], ones*3, label)
    label = torch.where(result_second[:, 0].logical_not() & result_first[:, 1].logical_not() & result_second[:, 1], ones*5, label)
    label = torch.where(result_first[:, 0] & result_second[:, 1].logical_not(), ones*6, label)
    label = torch.where(result_first[:, 0].logical_not() & result_second[:, 0] & result_second[:, 1].logical_not(), ones*7, label)
    label = torch.where(result_second[:, 0].logical_not() & result_second[:, 1].logical_not(), ones*8, label)

    criterion = nn.CrossEntropyLoss()
    loss = criterion(pred, label)

    pred = torch.argmax(pred, 1)
    corrects = (pred == label)
    acc = corrects.sum() / float(label.numel())
    return loss, acc


def compute_box_loss(data_dict, box_mask):
    """ Compute 3D bounding box loss.
    Args:
        data_dict: dict (read-only)
    Returns:
        center_loss
        size_reg_loss
    """

    # Compute center loss
    pred_center = data_dict['center']
    pred_size_residual = data_dict['size_residual']

    gt_center = data_dict['ref_center_label']
    gt_size_residual = data_dict['ref_size_residual_label']

    creterion = nn.SmoothL1Loss(reduction='none')
    center_loss = creterion(pred_center, gt_center)
    center_loss = (center_loss * box_mask.unsqueeze(1)).sum() / (box_mask.sum() + 1e-6)
    size_loss = creterion(pred_size_residual, gt_size_residual)
    size_loss = (size_loss * box_mask.unsqueeze(1)).sum() / (box_mask.sum() + 1e-6)

    return center_loss, size_loss


def compute_lang_classification_loss(data_dict):
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(data_dict["lang_scores"], data_dict["object_cat"])

    return loss


def get_loss(data_dict, config):
    """ Loss functions
    Args:
        data_dict: dict
        config: dataset config instance
        reference: flag (False/True)
    Returns:
        loss: pytorch scalar tensor
        data_dict: dict
    """
    lang_loss = compute_lang_classification_loss(data_dict)
    data_dict["lang_loss"] = lang_loss
    seg_loss, seg_acc = compute_scene_mask_loss(data_dict)

    # get ref gt
    ref_center_label = data_dict["ref_center_label"].detach().cpu().numpy()
    ref_heading_class_label = data_dict["ref_heading_class_label"].detach().cpu().numpy()
    ref_heading_residual_label = data_dict["ref_heading_residual_label"].detach().cpu().numpy()
    ref_size_class_label = data_dict["ref_size_class_label"].detach().cpu().numpy()
    ref_size_residual_label = data_dict["ref_size_residual_label"].detach().cpu().numpy()

    ref_gt_obb = config.param2obb_batch(ref_center_label, ref_heading_class_label, ref_heading_residual_label,
                                        ref_size_class_label, ref_size_residual_label)
    ref_gt_bbox = get_3d_box_batch(ref_gt_obb[:, 3:6], ref_gt_obb[:, 6], ref_gt_obb[:, 0:3])

    attribute_scores = data_dict['attribute_scores']
    relation_scores = data_dict['relation_scores']
    scene_scores = data_dict['scene_scores']

    pred_obb_batch = data_dict['pred_obb_batch']
    batch_size = len(pred_obb_batch)
    cluster_label = []
    box_mask = torch.zeros(batch_size).cuda()

    criterion = ContrastiveLoss(margin=0.2, gamma=5)
    ref_loss = torch.zeros(1).cuda().requires_grad_(True)
    start_idx = 0
    for i in range(batch_size):
        pred_obb = pred_obb_batch[i]  # (num, 7)
        num_filtered_obj = pred_obb.shape[0]
        if num_filtered_obj == 0:
            cluster_label.append([])
            box_mask[i] = 1
            continue

        label = np.zeros(num_filtered_obj)
        pred_bbox = get_3d_box_batch(pred_obb[:, 3:6], pred_obb[:, 6], pred_obb[:, 0:3])
        ious = box3d_iou_batch(pred_bbox, np.tile(ref_gt_bbox[i], (num_filtered_obj, 1, 1)))
        label[ious.argmax()] = 1  # treat the bbox with highest iou score as the gt

        label = torch.FloatTensor(label).cuda()
        cluster_label.append(label)
        if num_filtered_obj == 1: continue

        attribute_score = attribute_scores[start_idx:start_idx + num_filtered_obj]
        relation_score = relation_scores[start_idx:start_idx + num_filtered_obj]
        scene_score = scene_scores[start_idx:start_idx + num_filtered_obj]
        score = attribute_score + relation_score + scene_score

        start_idx += num_filtered_obj
        if ious.max() < 0.2: continue

        ref_loss = ref_loss + criterion(score, label)

    ref_loss = ref_loss / batch_size
    data_dict['ref_loss'] = ref_loss

    data_dict['loss'] = 10 * ref_loss + lang_loss + seg_loss
    data_dict["seg_loss"] = seg_loss
    data_dict['seg_acc'] = seg_acc
    data_dict['seg_loss'] = seg_loss
    data_dict['cluster_label'] = cluster_label

    return data_dict
