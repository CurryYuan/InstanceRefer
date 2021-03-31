import os
import sys
import argparse
import yaml

from easydict import EasyDict


def get_parser():
    parser = argparse.ArgumentParser(description='InstanceRefer')
    parser.add_argument('--gpu', type=str, default='0', help='GPU idx')
    parser.add_argument('--config', type=str, default='config/InstanceRefer.yaml', help='path to config file')
    parser.add_argument('--log_dir', type=str, default='test', help='path to log file')
    parser.add_argument('--debug', action='store_true')

    ### pretrain
    parser.add_argument('--pretrain', type=str, default='', help='path to pretrain model')

    args_cfg = parser.parse_args()
    assert args_cfg.config is not None
    with open(args_cfg.config, 'r') as f:
        config = yaml.load(f)

    for key in config:
        for k, v in config[key].items():
            setattr(args_cfg, k, v)

    return args_cfg


CONF = get_parser()

# path
CONF.PATH = EasyDict()
CONF.PATH.BASE = "/home/YOUR_PATH/InstanceRefer/"  # TODO: change this
CONF.PATH.DATA = os.path.join(CONF.PATH.BASE, "data")
CONF.PATH.SCANNET = os.path.join(CONF.PATH.DATA, "scannet")
CONF.PATH.LIB = os.path.join(CONF.PATH.BASE, "lib")
CONF.PATH.MODELS = os.path.join(CONF.PATH.BASE, "models")
CONF.PATH.UTILS = os.path.join(CONF.PATH.BASE, "utils")
CONF.exp_path = os.path.join(CONF.PATH.BASE, 'outputs', CONF.dataset, CONF.log_dir)
CONF.PATH.OUTPUT = os.path.join(CONF.exp_path, 'checkpoints')

# append to syspath
for _, path in CONF.PATH.items():
    sys.path.append(path)

# scannet data
CONF.PATH.SCANNET_SCANS = os.path.join(CONF.PATH.SCANNET, "scans")
CONF.PATH.SCANNET_META = os.path.join(CONF.PATH.SCANNET, "meta_data")
CONF.PATH.SCANNET_DATA = os.path.join(CONF.PATH.SCANNET, "pointgroup_data")

# data
CONF.SCANNET_DIR = "data/scannet/scans/"  # TODO change this
CONF.SCANNET_FRAMES_ROOT = "data/scanrefer/frames_square/"  # TODO change this
CONF.PROJECTION = "data/multiview_projection_scanrefer"  # TODO change this
CONF.ENET_FEATURES_ROOT = "data/scanrefer/enet_features/"  # TODO change this
CONF.ENET_FEATURES_SUBROOT = os.path.join(CONF.ENET_FEATURES_ROOT, "{}")  # scene_id
CONF.ENET_FEATURES_PATH = os.path.join(CONF.ENET_FEATURES_SUBROOT, "{}.npy")  # frame_id
CONF.SCANNET_FRAMES = os.path.join(CONF.SCANNET_FRAMES_ROOT, "{}/{}")  # scene_id, mode
CONF.SCENE_NAMES = sorted(os.listdir(CONF.SCANNET_DIR))
CONF.ENET_WEIGHTS = os.path.join(CONF.PATH.BASE, "data/scannetv2_enet.pth")
CONF.MULTIVIEW = os.path.join('data/', "enet_feats_maxpool.hdf5")
CONF.NYU40_LABELS = os.path.join(CONF.PATH.SCANNET_META, "nyu40_labels.csv")

# scannet
CONF.SCANNETV2_TRAIN = os.path.join(CONF.PATH.SCANNET_META, "scannetv2_train.txt")
CONF.SCANNETV2_VAL = os.path.join(CONF.PATH.SCANNET_META, "scannetv2_val.txt")
CONF.SCANNETV2_TEST = os.path.join(CONF.PATH.SCANNET_META, "scannetv2_test.txt")
CONF.SCANNETV2_LIST = os.path.join(CONF.PATH.SCANNET_META, "scannetv2.txt")

# train
CONF.TRAIN = EasyDict()
CONF.TRAIN.MAX_DES_LEN = 126
CONF.TRAIN.SEED = 42
