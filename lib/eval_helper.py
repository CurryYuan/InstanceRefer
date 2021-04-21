import os
import sys
import torch
import numpy as np

sys.path.append(os.path.join(os.getcwd(), "lib"))  # HACK add the lib folder
from utils.box_util import get_3d_box, box3d_iou
from utils.util import construct_bbox_corners


def get_eval(data_dict, config):
    """ Loss functions
    Args:
        data_dict: dict
        config: dataset config instance
        reference: flag (False/True)
        post_processing: config dict
    Returns:
        loss: pytorch scalar tensor
        data_dict: dict
    """
    lang_scores = data_dict["lang_scores"]
    lang_cls_pred = torch.argmax(lang_scores, dim=1)
    batch_size = lang_scores.shape[0]

    data_dict["lang_acc"] = (lang_cls_pred == data_dict["object_cat"]).float().mean()

    attribute_scores = data_dict['attribute_scores']
    relation_scores = data_dict['relation_scores']
    scene_scores = data_dict['scene_scores']

    pred_obb_batch = data_dict['pred_obb_batch']
    cluster_labels = data_dict['cluster_label']

    ref_center_label = data_dict["ref_center_label"].detach().cpu().numpy()
    ref_heading_class_label = data_dict["ref_heading_class_label"].detach().cpu().numpy()
    ref_heading_residual_label = data_dict["ref_heading_residual_label"].detach().cpu().numpy()
    ref_size_class_label = data_dict["ref_size_class_label"].detach().cpu().numpy()
    ref_size_residual_label = data_dict["ref_size_residual_label"].detach().cpu().numpy()

    ref_gt_obb = config.param2obb_batch(ref_center_label, ref_heading_class_label, ref_heading_residual_label,
                                        ref_size_class_label, ref_size_residual_label)

    ious = []
    pred_bboxes = []
    gt_bboxes = []
    ref_acc = []
    multiple = []
    others = []
    start_idx = 0
    num_missed = 0
    for i in range(batch_size):
        pred_obb = pred_obb_batch[i]  # (num, 7)
        num_filtered_obj = pred_obb.shape[0]
        if num_filtered_obj == 0:
            pred_obb = np.zeros(7)
            num_missed += 1
        elif num_filtered_obj == 1:
            pred_obb = pred_obb[0]
        else:
            attribute_score = attribute_scores[start_idx:start_idx + num_filtered_obj]
            relation_score = relation_scores[start_idx:start_idx + num_filtered_obj]
            scene_score = scene_scores[start_idx:start_idx + num_filtered_obj]
            score = attribute_score + relation_score + scene_score

            start_idx += num_filtered_obj
            cluster_pred = torch.argmax(score, dim=0)
            target = torch.argmax(cluster_labels[i], dim=0)
            if target == cluster_pred:
                ref_acc.append(1.)
            else:
                ref_acc.append(0.)

            pred_obb = pred_obb_batch[i][cluster_pred]

        gt_obb = ref_gt_obb[i]
        pred_bbox = get_3d_box(pred_obb[3:6], pred_obb[6], pred_obb[0:3])
        gt_bbox = get_3d_box(gt_obb[3:6], gt_obb[6], gt_obb[0:3])
        iou = box3d_iou(pred_bbox, gt_bbox)
        ious.append(iou)

        # NOTE: get_3d_box() will return problematic bboxes
        pred_bbox = construct_bbox_corners(pred_obb[0:3], pred_obb[3:6])
        gt_bbox = construct_bbox_corners(gt_obb[0:3], gt_obb[3:6])

        if num_filtered_obj <= 1:
            if iou > 0.25:
                ref_acc.append(1.)
            else:
                ref_acc.append(0.)

        pred_bboxes.append(pred_bbox)
        gt_bboxes.append(gt_bbox)

        # construct the multiple mask
        multiple.append(data_dict["unique_multiple"][i].item())

        # construct the others mask
        flag = 1 if data_dict["object_cat"][i] == 17 else 0
        others.append(flag)

    data_dict['ref_acc'] = ref_acc

    data_dict["ref_iou"] = ious
    data_dict["ref_iou_rate_0.25"] = np.array(ious)[np.array(ious) >= 0.25].shape[0] / np.array(ious).shape[0]
    data_dict["ref_iou_rate_0.5"] = np.array(ious)[np.array(ious) >= 0.5].shape[0] / np.array(ious).shape[0]

    # data_dict["seg_acc"] = torch.ones(1)[0].cuda()
    data_dict["ref_multiple_mask"] = multiple
    data_dict["ref_others_mask"] = others
    data_dict["pred_bboxes"] = pred_bboxes
    data_dict["gt_bboxes"] = gt_bboxes

    return data_dict