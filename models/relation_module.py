import torch
import torch.nn as nn
import numpy as np

from models.basic_blocks import DynamicEdgeConv

class RelationModule(nn.Module):
    def __init__(self, input_feature_dim, args, v_dim=128, h_dim=128, l_dim=256, dropout_rate=0.15):
        super().__init__()

        self.args = args
        self.input_feature_dim = input_feature_dim
        self.vis_emb_fc = nn.Sequential(nn.Linear(v_dim, h_dim),
                                        nn.LayerNorm(h_dim),
                                        nn.ReLU(),
                                        nn.Dropout(dropout_rate),
                                        nn.Linear(h_dim, h_dim),
                                        )

        self.lang_emb_fc = nn.Sequential(nn.Linear(l_dim, h_dim),
                                         nn.BatchNorm1d(h_dim),
                                         nn.ReLU(),
                                         nn.Dropout(dropout_rate),
                                         nn.Linear(h_dim, h_dim),
                                         )

        self.gcn = DynamicEdgeConv(input_feature_dim+args.num_classes, 128, k=args.k, num_classes=args.num_classes)
        self.one_hot_array = np.eye(args.num_classes)
        self.weight_initialization()


    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def filter_candidates(self, data_dict, lang_feats, lang_cls_pred):
        instance_points = data_dict['instance_points']
        pred_obb_batch = data_dict['pred_obb_batch']
        instance_obbs = data_dict['instance_obbs']
        batch_size = len(instance_points)

        batch_index = []
        lang_feats_flatten = []
        pred_obbs = []
        feats = []
        filtered_index = []

        for i in range(batch_size):
            num_filtered_obj = len(pred_obb_batch[i])
            if num_filtered_obj < 2:
                continue

            lang_feat = lang_feats[i]  # (1, h_dim)
            lang_feat = lang_feat.repeat(num_filtered_obj, 1)
            lang_feats_flatten.append(lang_feat)

            instance_point = instance_points[i]
            instance_obb = instance_obbs[i]
            instance_class = data_dict['instance_class'][i]
            num_obj = len(instance_point)
            pred_obbs += list(instance_obb)

            # filter by class
            for j in range(num_obj):
                point_cloud = instance_point[j]
                point_cloud = point_cloud.mean(0)
                point_cloud[:3] = instance_obb[j][:3]

                onhot_semantic = self.one_hot_array[instance_class[j]]
                point_cloud = np.concatenate([point_cloud, onhot_semantic], -1)
                feats.append(point_cloud)
                if instance_class[j] == lang_cls_pred[i]:
                    filtered_index.append(len(batch_index))
                batch_index.append(i)

        return feats, lang_feats_flatten, batch_index, filtered_index, pred_obbs

    def forward(self, data_dict):
        lang_feats = data_dict['lang_rel_feats']  # (B, l_dim)
        lang_feats = self.lang_emb_fc(lang_feats).unsqueeze(1)  # (B, 1, h_dim)

        if not self.args.use_gt_lang:
            lang_scores = data_dict["lang_scores"]
            lang_cls_pred = torch.argmax(lang_scores, dim=1)
        else:
            lang_cls_pred = data_dict['object_cat']

        feats, lang_feats_flatten, batch_index, filtered_index, pred_obbs = \
            self.filter_candidates(data_dict, lang_feats, lang_cls_pred)

        lang_feats_flatten = torch.cat(lang_feats_flatten, dim=0)
        feats = torch.Tensor(feats).cuda()

        batch_index = torch.LongTensor(batch_index).cuda()
        filtered_index = torch.LongTensor(filtered_index).cuda()
        support_xyz = torch.Tensor(pred_obbs)[:, :3].cuda()

        feats = self.gcn(support_xyz, batch_index, filtered_index, feats)
        feats = self.vis_emb_fc(feats)

        scores = nn.functional.cosine_similarity(feats, lang_feats_flatten, dim=1)

        data_dict['relation_scores'] = scores

        return data_dict



