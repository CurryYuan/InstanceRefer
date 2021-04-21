import math
import torch
import torch.nn as nn
import numpy as np
import torchsparse.nn as spnn

from models.basic_blocks import BEVEncoder, SparseCrop, ToDenseBEVConvolution

class SceneModule(nn.Module):
    def __init__(self, input_feature_dim, args, v_dim=128, h_dim=128, l_dim=256, dropout_rate=0.15):
        super().__init__()

        self.args = args
        self.input_feature_dim = input_feature_dim
        self.voxel_size = np.array([args.voxel_size_glp]*3)

        # Sparse Volumetric Backbone
        self.net = BEVEncoder(self.input_feature_dim)

        self.pooling = spnn.GlobalMaxPooling()

        loc_max = torch.tensor([240, 400, 80], device='cuda', dtype=torch.int32)
        loc_min = torch.tensor([0, 0, 0], device='cuda', dtype=torch.int32)

        self.to_bev = nn.Sequential(
            SparseCrop(loc_min=loc_min, loc_max=loc_max),
            ToDenseBEVConvolution(128, 128, shape=(loc_max-loc_min)//16, z_dim=2, offset=loc_min),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )

        self.h_dim = h_dim
        self.vis_emb_fc = nn.Sequential(nn.Conv2d(v_dim, h_dim, 3),
                                        nn.BatchNorm2d(h_dim),
                                        nn.ReLU(),
                                        nn.Dropout(dropout_rate),
                                        nn.Conv2d(h_dim, h_dim, 3),
                                        )

        self.vis_emb_fc1 = nn.Sequential(nn.Linear(128, h_dim),
                                        nn.LayerNorm(h_dim),
                                        nn.ReLU(),
                                        nn.Dropout(dropout_rate),
                                        nn.Linear(h_dim, h_dim),
                                        )

        self.lang_emb_fc = nn.Sequential(nn.Linear(l_dim, h_dim),
                                         nn.LayerNorm(h_dim),
                                         nn.ReLU(),
                                         nn.Dropout(dropout_rate),
                                         nn.Linear(h_dim, h_dim),
                                         )

        self.cls = nn.Sequential(nn.Linear(h_dim, h_dim),
                                 nn.BatchNorm1d(h_dim),
                                 nn.ReLU(),
                                 nn.Linear(h_dim, 9),
                                 )

    def forward(self, data_dict):
        feats = data_dict['lidar']
        point_min = data_dict['point_min']
        batch_size = point_min.shape[0]
        pred_obb_batch = data_dict['pred_obb_batch']
        obj_feats_flatten = data_dict['obj_feats']
        lang_feats = data_dict['lang_scene_feats']

        # Sparse Volumetric Backbone
        feats = self.net(feats)
        feats = self.to_bev(feats)  # BCHW
        feats = self.vis_emb_fc(feats)  # (B, D, H, W)

        h, w = feats.shape[-2:]
        feats = feats.reshape(batch_size, self.h_dim, -1).permute(0, 2, 1)  # (B, n_vis, dim)
        lang_feats = self.lang_emb_fc(lang_feats).unsqueeze(2)

        atten = torch.bmm(feats, lang_feats) / math.sqrt(feats.shape[2])  # (B, n_vis, n_lang)
        atten = atten.squeeze(2)
        atten = torch.softmax(atten, dim=1)
        data_dict['vis_atten'] = atten.reshape(batch_size, h, w)

        # cls
        scene_feats = torch.sum(feats * atten.unsqueeze(2), dim=1)
        seg_scores = self.cls(scene_feats)

        data_dict['seg_scores'] = seg_scores

        # matching
        scene_feats_flatten = []

        for i in range(batch_size):
            num_filtered_obj = len(pred_obb_batch[i])
            if num_filtered_obj < 2:
                continue

            scene_feat = scene_feats[i]  # (1, h_dim)
            scene_feat = scene_feat.repeat(num_filtered_obj, 1)
            scene_feats_flatten.append(scene_feat)

        scene_feats_flatten = torch.cat(scene_feats_flatten, dim=0)

        # L2 Normalize
        obj_feats_flatten = self.vis_emb_fc1(obj_feats_flatten)
        scores = nn.functional.cosine_similarity(obj_feats_flatten, scene_feats_flatten, dim=1)

        data_dict['scene_scores'] = scores

        return data_dict

