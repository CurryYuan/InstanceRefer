import os
import sys
import json
import pickle
import numpy as np

import torch
from torch.utils.data import DataLoader

from tqdm import tqdm

sys.path.append(os.path.join(os.getcwd()))  # HACK add the root folder
from lib.config import CONF
from lib.dataset import ScannetReferenceDataset
from lib.loss_helper import get_loss
from lib.eval_helper import get_eval
from data.scannet.model_util_scannet import ScannetDatasetConfig
from models.instancerefer import InstanceRefer

SCANREFER_TRAIN = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_train.json")))
SCANREFER_VAL = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_val.json")))


def get_dataloader(args, scanrefer, all_scene_list, split):
    dataset = ScannetReferenceDataset(
        scanrefer=scanrefer,
        scanrefer_all_scene=all_scene_list,
        split=split,
        args=CONF
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=dataset.collate_fn
    )

    return dataset, dataloader


def get_model(args):
    # load model
    input_channels = int(args.use_multiview) * 128 + int(args.use_normal) * 3 + \
                     int(args.use_color) * 3 + int(args.use_height + 3)

    model = InstanceRefer(
        input_feature_dim=input_channels,
        args=args
    )

    path = os.path.join(args.use_checkpoint, "model_last.pth")
    model.load_state_dict(torch.load(path))
    model.eval()
    model.cuda()

    return model


def get_scannet_scene_list(split):
    scene_list = sorted(
        [line.rstrip() for line in open(os.path.join(CONF.PATH.SCANNET_META, "scannetv2_{}.txt".format(split)))])

    return scene_list


def get_scanrefer():
    scene_list = sorted(list(set([data["scene_id"] for data in SCANREFER_VAL])))
    scanrefer = [data for data in SCANREFER_VAL if data["scene_id"] in scene_list]

    return scanrefer, scene_list


def eval_ref(args):
    print("evaluate...")
    # constant
    DC = ScannetDatasetConfig()

    # init training dataset
    print("preparing data...")
    scanrefer, scene_list = get_scanrefer()

    # dataloader
    _, dataloader = get_dataloader(args, scanrefer, scene_list, "val")

    # model
    model = get_model(args)

    # random seeds
    seeds = [args.manual_seed]

    # evaluate
    print("evaluating...")
    score_path = os.path.join(args.use_checkpoint, "scores.p")
    pred_path = os.path.join(args.use_checkpoint, "predictions.p")

    if not os.path.exists(score_path):
        ref_acc_all = []
        ious_all = []
        masks_all = []
        others_all = []
        lang_acc_all = []
        for seed in seeds:
            # reproducibility
            torch.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            np.random.seed(seed)

            print("generating the scores for seed {}...".format(seed))
            ref_acc = []
            ious = []
            masks = []
            others = []
            lang_acc = []
            predictions = {}
            for data in tqdm(dataloader):
                for key in data:
                    if key in ['lang_feat', 'lang_len', 'object_cat', 'lidar', 'point_min', 'point_max', 'mlm_label',
                               'ref_center_label', 'ref_size_residual_label']:
                        data[key] = data[key].cuda()

                # feed
                data = model(data)
                data = get_loss(data, DC)
                data = get_eval(
                    data_dict=data,
                    config=DC,
                )

                ref_acc += data["ref_acc"]
                ious += data["ref_iou"]
                masks += data["ref_multiple_mask"]
                others += data["ref_others_mask"]
                lang_acc.append(data["lang_acc"].item())

                # store predictions
                ids = data["scan_idx"].detach().cpu().numpy()
                for i in range(ids.shape[0]):
                    idx = ids[i]
                    scene_id = scanrefer[idx]["scene_id"]
                    object_id = scanrefer[idx]["object_id"]
                    ann_id = scanrefer[idx]["ann_id"]

                    if scene_id not in predictions:
                        predictions[scene_id] = {}

                    if object_id not in predictions[scene_id]:
                        predictions[scene_id][object_id] = {}

                    if ann_id not in predictions[scene_id][object_id]:
                        predictions[scene_id][object_id][ann_id] = {}

                    predictions[scene_id][object_id][ann_id][""] = data["pred_bboxes"][i]
                    predictions[scene_id][object_id][ann_id]["gt_bpred_bboxbox"] = data["gt_bboxes"][i]
                    predictions[scene_id][object_id][ann_id]["iou"] = data["ref_iou"][i]

            # save the last predictions
            with open(pred_path, "wb") as f:
                pickle.dump(predictions, f)

            # save to global
            ref_acc_all.append(ref_acc)
            ious_all.append(ious)
            masks_all.append(masks)
            others_all.append(others)
            lang_acc_all.append(lang_acc)

        # convert to numpy array
        ref_acc = np.array(ref_acc_all)
        ious = np.array(ious_all)
        masks = np.array(masks_all)
        others = np.array(others_all)
        lang_acc = np.array(lang_acc_all)

        # save the global scores
        with open(score_path, "wb") as f:
            scores = {
                "ref_acc": ref_acc_all,
                "ious": ious_all,
                "masks": masks_all,
                "others": others_all,
                "lang_acc": lang_acc_all
            }
            pickle.dump(scores, f)

    else:
        print("loading the scores...")
        with open(score_path, "rb") as f:
            scores = pickle.load(f)

            # unpack
            ref_acc = np.array(scores["ref_acc"])
            ious = np.array(scores["ious"])
            masks = np.array(scores["masks"])
            others = np.array(scores["others"])
            lang_acc = np.array(scores["lang_acc"])

    multiple_dict = {
        "unique": 0,
        "multiple": 1
    }
    others_dict = {
        "not_in_others": 0,
        "in_others": 1
    }

    # evaluation stats
    stats = {k: np.sum(masks[0] == v) for k, v in multiple_dict.items()}
    stats["overall"] = masks[0].shape[0]
    stats = {}
    for k, v in multiple_dict.items():
        stats[k] = {}
        for k_o, v_o in others_dict.items():
            stats[k][k_o] = np.sum(np.logical_and(masks[0] == v, others[0] == v_o))

        stats[k]["overall"] = np.sum(masks[0] == v)

    stats["overall"] = {}
    for k_o, v_o in others_dict.items():
        stats["overall"][k_o] = np.sum(others[0] == v_o)

    stats["overall"]["overall"] = masks[0].shape[0]

    # aggregate scores
    scores = {}
    for k, v in multiple_dict.items():
        for k_o in others_dict.keys():
            ref_accs, acc_025ious, acc_05ious = [], [], []
            for i in range(masks.shape[0]):
                running_ref_acc = np.mean(
                    ref_acc[i][np.logical_and(masks[i] == multiple_dict[k], others[i] == others_dict[k_o])]) \
                    if np.sum(np.logical_and(masks[i] == multiple_dict[k], others[i] == others_dict[k_o])) > 0 else 0
                running_acc_025iou = ious[i][np.logical_and(
                    np.logical_and(masks[i] == multiple_dict[k], others[i] == others_dict[k_o]),
                    ious[i] >= 0.25)].shape[0] \
                                     / ious[i][np.logical_and(masks[i] == multiple_dict[k],
                                                              others[i] == others_dict[k_o])].shape[0] \
                    if np.sum(np.logical_and(masks[i] == multiple_dict[k], others[i] == others_dict[k_o])) > 0 else 0
                running_acc_05iou = ious[i][np.logical_and(
                    np.logical_and(masks[i] == multiple_dict[k], others[i] == others_dict[k_o]), ious[i] >= 0.5)].shape[0] \
                                    / ious[i][np.logical_and(masks[i] == multiple_dict[k],
                                                             others[i] == others_dict[k_o])].shape[0] \
                    if np.sum(np.logical_and(masks[i] == multiple_dict[k], others[i] == others_dict[k_o])) > 0 else 0

                # store
                ref_accs.append(running_ref_acc)
                acc_025ious.append(running_acc_025iou)
                acc_05ious.append(running_acc_05iou)

            if k not in scores:
                scores[k] = {k_o: {} for k_o in others_dict.keys()}

            scores[k][k_o]["ref_acc"] = np.mean(ref_accs)
            scores[k][k_o]["acc@0.25iou"] = np.mean(acc_025ious)
            scores[k][k_o]["acc@0.5iou"] = np.mean(acc_05ious)

        ref_accs, acc_025ious, acc_05ious = [], [], []
        for i in range(masks.shape[0]):
            running_ref_acc = np.mean(ref_acc[i][masks[i] == multiple_dict[k]]) if np.sum(
                masks[i] == multiple_dict[k]) > 0 else 0
            running_acc_025iou = ious[i][np.logical_and(masks[i] == multiple_dict[k], ious[i] >= 0.25)].shape[0] \
                                 / ious[i][masks[i] == multiple_dict[k]].shape[0] if np.sum(
                masks[i] == multiple_dict[k]) > 0 else 0
            running_acc_05iou = ious[i][np.logical_and(masks[i] == multiple_dict[k], ious[i] >= 0.5)].shape[0] \
                                / ious[i][masks[i] == multiple_dict[k]].shape[0] if np.sum(
                masks[i] == multiple_dict[k]) > 0 else 0

            # store
            ref_accs.append(running_ref_acc)
            acc_025ious.append(running_acc_025iou)
            acc_05ious.append(running_acc_05iou)

        scores[k]["overall"] = {}
        scores[k]["overall"]["ref_acc"] = np.mean(ref_accs)
        scores[k]["overall"]["acc@0.25iou"] = np.mean(acc_025ious)
        scores[k]["overall"]["acc@0.5iou"] = np.mean(acc_05ious)

    scores["overall"] = {}
    for k_o in others_dict.keys():
        ref_accs, acc_025ious, acc_05ious = [], [], []
        for i in range(masks.shape[0]):
            running_ref_acc = np.mean(ref_acc[i][others[i] == others_dict[k_o]]) if np.sum(
                others[i] == others_dict[k_o]) > 0 else 0
            running_acc_025iou = ious[i][np.logical_and(others[i] == others_dict[k_o], ious[i] >= 0.25)].shape[0] \
                                 / ious[i][others[i] == others_dict[k_o]].shape[0] if np.sum(
                others[i] == others_dict[k_o]) > 0 else 0
            running_acc_05iou = ious[i][np.logical_and(others[i] == others_dict[k_o], ious[i] >= 0.5)].shape[0] \
                                / ious[i][others[i] == others_dict[k_o]].shape[0] if np.sum(
                others[i] == others_dict[k_o]) > 0 else 0

            # store
            ref_accs.append(running_ref_acc)
            acc_025ious.append(running_acc_025iou)
            acc_05ious.append(running_acc_05iou)

        # aggregate
        scores["overall"][k_o] = {}
        scores["overall"][k_o]["ref_acc"] = np.mean(ref_accs)
        scores["overall"][k_o]["acc@0.25iou"] = np.mean(acc_025ious)
        scores["overall"][k_o]["acc@0.5iou"] = np.mean(acc_05ious)

    ref_accs, acc_025ious, acc_05ious = [], [], []
    for i in range(masks.shape[0]):
        running_ref_acc = np.mean(ref_acc[i])
        running_acc_025iou = ious[i][ious[i] >= 0.25].shape[0] / ious[i].shape[0]
        running_acc_05iou = ious[i][ious[i] >= 0.5].shape[0] / ious[i].shape[0]

        # store
        ref_accs.append(running_ref_acc)
        acc_025ious.append(running_acc_025iou)
        acc_05ious.append(running_acc_05iou)

    # aggregate
    scores["overall"]["overall"] = {}
    scores["overall"]["overall"]["ref_acc"] = np.mean(ref_accs)
    scores["overall"]["overall"]["acc@0.25iou"] = np.mean(acc_025ious)
    scores["overall"]["overall"]["acc@0.5iou"] = np.mean(acc_05ious)

    # report
    print("\nstats:")
    for k_s in stats.keys():
        for k_o in stats[k_s].keys():
            print("{} | {}: {}".format(k_s, k_o, stats[k_s][k_o]))

    for k_s in scores.keys():
        print("\n{}:".format(k_s))
        for k_m in scores[k_s].keys():
            for metric in scores[k_s][k_m].keys():
                print("{} | {} | {}: {:.4f}".format(k_s, k_m, metric, scores[k_s][k_m][metric]))

    print("\nlanguage classification accuracy: {:.4f}".format(np.mean(lang_acc)))


if __name__ == "__main__":
    # setting
    os.environ["CUDA_VISIBLE_DEVICES"] = CONF.gpu

    # evaluate
    eval_ref(CONF)
