import numpy as np

from models import util
from models.basic_blocks import *
from torchsparse import SparseTensor
from torchsparse.utils import sparse_quantize, sparse_collate, sparse_collate_tensors


class AttributeModule(nn.Module):
    def __init__(self, input_feature_dim, args):
        super().__init__()

        self.args = args
        self.use_semantic = args.use_semantic
        self.instance_augment = args.instance_augment
        self.input_feature_dim = input_feature_dim

        v_dim = args.visual_dim
        l_dim = args.languege_dim
        h_dim = args.hidden_dim

        # Sparse Volumetric Backbone
        self.net = SparseEncoder(self.input_feature_dim, v_dim)
        self.pooling = spnn.GlobalMaxPooling()

        self.voxel_size = np.array([args.voxel_size_ap, args.voxel_size_ap, args.voxel_size_ap])

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

    def instance_augmentor(self, pc, dropout=True, scaling=False, jitter=False, shuffle=False, rotation=True):

        '''Data augmentation'''
        pc = pc - pc.mean(0)
        if dropout:
            pc = util.random_point_dropout(pc)
        if scaling:
            pc = util.random_scale_point_cloud(pc)
        if jitter:
            pc = util.jitter_point_cloud(pc)
        if shuffle:
            pc = util.shuffle_points(pc)
        if rotation:
            pc = util.rotate_point_cloud_z(pc)

        return pc

    def one_hot(self, length, position):
        zeros = [0 for _ in range(length)]
        zeros[position] = 1
        zeros = np.array(zeros)
        return zeros

    def filter_candidates(self, data_dict, lang_cls_pred):
        pred_obb_batch = []
        pts_batch = []
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
                    point_cloud = instance_point[j]
                    onhot_semantic = self.one_hot(self.args.num_classes, lang_cls_pred[i])

                    pc = point_cloud[:, :3]

                    if self.training and self.instance_augment:
                        pc = self.instance_augmentor(pc)

                    pred_obbs.append(instance_obb[j])

                    if self.use_semantic:
                        rep_cls = np.tile(onhot_semantic, [point_cloud.shape[0], 1])
                        point_cloud = np.concatenate([point_cloud, rep_cls], -1)

                    coords, feats = sparse_quantize(
                        pc,
                        point_cloud,
                        quantization_size=self.voxel_size
                    )
                    pt = SparseTensor(feats, coords)
                    pts.append(pt)

            if len(pts) == 1:
                pts = []
            pts_batch += pts
            pred_obbs = np.asarray(pred_obbs)
            pred_obb_batch.append(pred_obbs)

        return pts_batch, pred_obb_batch

    def forward(self, data_dict):
        batch_size = len(data_dict['instance_points'])

        # lang encoding
        lang_feats = data_dict['lang_attr_feats']  # (B, l_dim)
        lang_feats = self.lang_emb_fc(lang_feats)  # (B, h_dim)
        lang_feats = nn.functional.normalize(lang_feats, p=2, dim=1).unsqueeze(1)  # (B, 1, h_dim)

        # filter candidates
        if not self.args.use_gt_lang:
            lang_scores = data_dict["lang_scores"]
            lang_cls_pred = torch.argmax(lang_scores, dim=1)
            pts_batch, pred_obb_batch = self.filter_candidates(data_dict, lang_cls_pred)

        else:
            pts_batch = data_dict['pts_batch']
            pred_obb_batch = data_dict['pred_obb_batch']

        feats = sparse_collate_tensors(pts_batch).cuda()

        # feature extractor
        feats = self.net(feats)
        feats = self.pooling(feats)  # (num_filtered_obj, dim)
        feats = self.vis_emb_fc(feats)

        lang_feats_flatten = []
        for i in range(batch_size):
            num_filtered_obj = len(pred_obb_batch[i])

            if num_filtered_obj < 2:
                continue

            lang_feat = lang_feats[i]  # (1, h_dim)
            lang_feat = lang_feat.repeat(num_filtered_obj, 1)
            lang_feats_flatten.append(lang_feat)

        lang_feats_flatten = torch.cat(lang_feats_flatten, dim=0)
        # feats = nn.functional.normalize(feats, p=2, dim=1)
        scores = torch.sum(feats * lang_feats_flatten, dim=1)

        data_dict['obj_feats'] = feats
        data_dict['attribute_scores'] = scores
        data_dict['pred_obb_batch'] = pred_obb_batch

        return data_dict
