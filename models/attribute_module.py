import torch
import torch.nn as nn
import numpy as np
import torchsparse.nn as spnn

from models.basic_blocks import SparseConvEncoder
from torchsparse import SparseTensor
from torchsparse.utils import sparse_quantize, sparse_collate_tensors


class AttributeModule(nn.Module):
    def __init__(self, input_feature_dim, args, v_dim=128, h_dim=256, l_dim=256):
        super().__init__()
        self.args = args
        self.input_feature_dim = input_feature_dim
        self.voxel_size = np.array([args.voxel_size_ap]*3)

        # Sparse Volumetric Backbone
        self.net = SparseConvEncoder(self.input_feature_dim)
        self.pooling = spnn.GlobalMaxPooling()

        self.vis_emb_fc = nn.Sequential(nn.Linear(v_dim, h_dim),
                                        nn.LayerNorm(h_dim),
                                        nn.ReLU(),
                                        nn.Linear(h_dim, h_dim),
                                        )

        self.lang_emb_fc = nn.Sequential(nn.Linear(l_dim, h_dim),
                                         nn.BatchNorm1d(h_dim),
                                         nn.ReLU(),
                                         nn.Linear(h_dim, h_dim),
                                         )

        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def filter_candidates(self, data_dict, lang_cls_pred):
        pred_obb_batch = []
        pts_batch = []
        obj_points_batch = []
        num_filtered_objs = []
        batch_size = len(data_dict['instance_points'])

        for i in range(batch_size):
            instance_point = data_dict['instance_points'][i]
            instance_obb = data_dict['instance_obbs'][i]
            instance_class = data_dict['instance_class'][i]
            num_obj = len(instance_point)

            pts = []
            pred_obbs = []

            # filter by class
            for j in range(num_obj):
                if instance_class[j] == lang_cls_pred[i]:
                    pred_obbs.append(instance_obb[j])
                    point_cloud = instance_point[j]
                    pc = point_cloud[:, :3]

                    coords, feats = sparse_quantize(
                        pc,
                        point_cloud,
                        quantization_size=self.voxel_size
                    )
                    pt = SparseTensor(feats, coords)
                    pts.append(pt)
                    obj_points_batch.append(point_cloud)

            num_filtered_objs.append(len(pts))
            if len(pts) < 2:
                pts = []
            pts_batch += pts
            pred_obbs = np.asarray(pred_obbs)
            pred_obb_batch.append(pred_obbs)

        return pts_batch, pred_obb_batch, num_filtered_objs

    def forward(self, data_dict):
        instance_points = data_dict['instance_points']
        batch_size = len(instance_points)

        # lang encoding
        lang_feats = data_dict['lang_attr_feats']  # (B, l_dim)
        lang_feats = self.lang_emb_fc(lang_feats)  # (B, h_dim)
        lang_feats = nn.functional.normalize(lang_feats, p=2, dim=1).unsqueeze(1)  # (B, 1, h_dim)

        # filter candidates
        if not self.args.use_gt_lang:
            lang_scores = data_dict["lang_scores"]
            lang_cls_pred = torch.argmax(lang_scores, dim=1)
        else:
            lang_cls_pred = data_dict['object_cat']

        pts_batch, pred_obb_batch, num_filtered_objs = self.filter_candidates(data_dict, lang_cls_pred)
        data_dict['num_filtered_objs'] = num_filtered_objs
        feats = sparse_collate_tensors(pts_batch).cuda()

        # Sparse Volumetric Backbone
        feats = self.net(feats)
        feats = self.pooling(feats)  # (num_filtered_obj, dim)
        data_dict['obj_feats'] = feats

        feats = self.vis_emb_fc(feats)

        data_dict['num_filtered_objs'] = num_filtered_objs

        # L2 Normalize
        feats = nn.functional.normalize(feats, p=2, dim=1)

        lang_feats_flatten = []
        for i in range(batch_size):
            num_filtered_obj = len(pred_obb_batch[i])
            if num_filtered_obj < 2:
                continue

            lang_feat = lang_feats[i]  # (1, h_dim)
            lang_feat = lang_feat.repeat(num_filtered_obj, 1)
            lang_feats_flatten.append(lang_feat)

        lang_feats_flatten = torch.cat(lang_feats_flatten, dim=0)
        scores = torch.sum(feats * lang_feats_flatten, dim=1)

        data_dict['attribute_scores'] = scores
        data_dict['pred_obb_batch'] = pred_obb_batch

        return data_dict


