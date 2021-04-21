import os
import sys
import json
import torch
import random

import numpy as np
import torch.optim as optim

sys.path.append("../utils")  # HACK add the lib folder
sys.path.append(os.path.join(os.getcwd()))  # HACK add the root folder

from torch.utils.data import DataLoader
from data.scannet.model_util_scannet import ScannetDatasetConfig
from lib.dataset import ScannetReferenceDataset
from lib.solver import Solver
from lib.config import CONF
from models.instancerefer import InstanceRefer

SCANREFER_TRAIN = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_train.json")))
SCANREFER_VAL = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_val.json")))

# constants
DC = ScannetDatasetConfig()

def init():
    # copy important files to backup
    backup_dir = os.path.join(CONF.exp_path, 'backup_files')
    os.makedirs(backup_dir, exist_ok=True)
    os.system('cp {}/scripts/train.py {}'.format(CONF.PATH.BASE, backup_dir))
    os.system('cp {} {}'.format(CONF.config, backup_dir))
    os.system('cp {} {}'.format(CONF.PATH.BASE+'/models/util.py', backup_dir))
    os.system('cp {}/models/{}.py {}'.format(CONF.PATH.BASE, CONF.model, backup_dir))
    os.system('cp {}/models/{}.py {}'.format(CONF.PATH.BASE, CONF.language_module, backup_dir))

    if CONF.attribute_module:
        os.system('cp {}/models/{}.py {}'.format(CONF.PATH.BASE, CONF.attribute_module, backup_dir))
    if CONF.relation_module:
        os.system('cp {}/models/{}.py {}'.format(CONF.PATH.BASE, CONF.relation_module, backup_dir))
    if CONF.scene_module:
        os.system('cp {}/models/{}.py {}'.format(CONF.PATH.BASE, CONF.scene_module, backup_dir))

    # random seed
    random.seed(CONF.manual_seed)
    np.random.seed(CONF.manual_seed)
    torch.manual_seed(CONF.manual_seed)
    torch.cuda.manual_seed_all(CONF.manual_seed)
    torch.backends.cudnn.benchmark = True
    np.random.seed(CONF.manual_seed)


def get_dataloader(args, scanrefer, all_scene_list, split, shuffle):
    dataset = ScannetReferenceDataset(
        scanrefer=scanrefer[split],
        scanrefer_all_scene=all_scene_list,
        split=split,
        args=CONF
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=dataset.collate_fn
    )

    return dataset, dataloader


def get_model(args):
    # initiate model
    input_channels = int(args.use_multiview) * 128 + int(args.use_normal) * 3 +\
                     int(args.use_color) * 3 + int(args.use_height + 3)

    model = InstanceRefer(
        input_feature_dim=input_channels,
        args=CONF
    )

    # trainable model
    if args.use_pretrained:
        # load model
        print("loading pretrained model...")
        pretrained_model = InstanceRefer(
            input_feature_dim=input_channels,
            args=CONF
        )

        pretrained_path = os.path.join(args.use_pretrained, "model.pth")
        pretrained_model.load_state_dict(torch.load(pretrained_path), strict=False)
        model.lang = pretrained_model.lang
        model.attribute = pretrained_model.attribute
        model.relation = pretrained_model.relation
        model.scene = pretrained_model.scene

    # to CUDA
    model = model.cuda()
    return model


def get_num_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_params = int(sum([np.prod(p.size()) for p in model_parameters]))

    return num_params


def get_solver(args, dataloader):
    model = get_model(args)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    if args.use_checkpoint:
        print("loading checkpoint {}...".format(args.use_checkpoint))
        stamp = args.use_checkpoint
        checkpoint = torch.load(args.use_checkpoint)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    else:
        stamp = CONF.PATH.OUTPUT
        os.makedirs(stamp, exist_ok=True)

    solver = Solver(
        model=model,
        config=DC,
        dataloader=dataloader,
        optimizer=optimizer,
        stamp=stamp,
        val_step=args.val_step,
        lr_decay_step=args.lr_decay_step,
        lr_decay_rate=args.lr_decay_rate,
        bn_decay_step=args.bn_decay_step,
        bn_decay_rate=args.bn_decay_rate,
    )

    num_params = get_num_params(model)
    print('model params', num_params)

    return solver, num_params, stamp


def save_info(args, root, num_params, train_dataset, val_dataset):
    info = {}
    for key, value in vars(args).items():
        info[key] = value

    info["num_train"] = len(train_dataset)
    info["num_val"] = len(val_dataset)
    info["num_train_scenes"] = len(train_dataset.scene_list)
    info["num_val_scenes"] = len(val_dataset.scene_list)
    info["num_params"] = num_params

    with open(os.path.join(root, "info.json"), "w") as f:
        json.dump(info, f, indent=4)


def get_scannet_scene_list(split):
    scene_list = sorted(
        [line.rstrip() for line in open(os.path.join(CONF.PATH.SCANNET_META, "scannetv2_{}.txt".format(split)))])

    return scene_list


def get_scanrefer(scanrefer_train, scanrefer_val, num_scenes):
    # get initial scene list
    train_scene_list = sorted(list(set([data["scene_id"] for data in scanrefer_train])))
    val_scene_list = sorted(list(set([data["scene_id"] for data in scanrefer_val])))
    if num_scenes == -1:
        num_scenes = len(train_scene_list)
    else:
        assert len(train_scene_list) >= num_scenes

    # slice train_scene_list
    train_scene_list = train_scene_list[:num_scenes]

    # filter data in chosen scenes
    new_scanrefer_train = []
    for data in scanrefer_train:
        if data["scene_id"] in train_scene_list:
            new_scanrefer_train.append(data)

    new_scanrefer_val = scanrefer_val

    # all scanrefer scene
    all_scene_list = train_scene_list + val_scene_list

    print("train on {} samples and val on {} samples".format(len(new_scanrefer_train), len(new_scanrefer_val)))

    return new_scanrefer_train, new_scanrefer_val, all_scene_list


def train(args):
    # init training dataset
    print("Preparing data...")
    scanrefer_train, scanrefer_val, all_scene_list = get_scanrefer(SCANREFER_TRAIN, SCANREFER_VAL, args.num_scenes)
    scanrefer = {"train": scanrefer_train, "val": scanrefer_val}

    # dataloader
    train_dataset, train_dataloader = get_dataloader(args, scanrefer, all_scene_list, "train", shuffle=True)
    val_dataset, val_dataloader = get_dataloader(args, scanrefer, all_scene_list, "val", shuffle=False)

    dataloader = {
        "train": train_dataloader,
        "val": val_dataloader
    }

    print("Initializing...")
    solver, num_params, root = get_solver(args, dataloader)

    print("Start training...\n")
    save_info(args, root, num_params, train_dataset, val_dataset)
    solver(args.epoch, args.verbose)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = CONF.gpu

    if CONF.debug:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    print('Use GPU:', torch.cuda.is_available())
    init()
    train(CONF)


