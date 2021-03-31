"""Util"""
import torch
import numpy as np
import torch.nn as nn
import torch.nn.init

class Util:

    @staticmethod
    def l1norm(X, dim, eps=1e-8):
        """L1-normalize columns of X
        """
        norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
        X = torch.div(X, norm)
        return X

    @staticmethod
    def l2norm(X, dim, eps=1e-8):
        """L2-normalize columns of X
        """
        norm = torch.pow(X, 2).sum(dim=dim, keepdim=True)
        norm = torch.sqrt(norm + eps)
        X = torch.div(X, norm)
        return X

    @staticmethod
    def cosine_similarity(x1, x2, dim=1, eps=1e-8):
        """Returns cosine similarity between x1 and x2, computed along dim."""
        w12 = torch.sum(x1 * x2, dim)
        w1 = torch.norm(x1, 2, dim)
        w2 = torch.norm(x2, 2, dim)
        return (w12 / (w1 * w2).clamp(min=eps)).squeeze()

    @staticmethod
    def inter_relation(K, Q, xlambda):
        """
        Q: (batch, queryL, d)
        K: (batch, sourceL, d)
        return (batch, queryL, sourceL)
        """
        batch_size, queryL = Q.size(0), Q.size(1)
        batch_size, sourceL = K.size(0), K.size(1)

        # (batch, sourceL, d)(batch, d, queryL)
        # --> (batch, sourceL, queryL)
        queryT = torch.transpose(Q, 1, 2)

        attn = torch.bmm(K, queryT)
        attn = nn.LeakyReLU(0.1)(attn)
        attn = Util.l2norm(attn, 2)

        # --> (batch, queryL, sourceL)
        attn = torch.transpose(attn, 1, 2).contiguous()
        # --> (batch*queryL, sourceL)
        attn = attn.view(batch_size * queryL, sourceL)
        attn = nn.Softmax(dim=1)(attn * xlambda)
        # --> (batch, queryL, sourceL)
        attn = attn.view(batch_size, queryL, sourceL)
        # --> (batch, sourceL, queryL)
        return attn

    @staticmethod
    def intra_relation(K, Q, xlambda):
        """
        Q: (n_context, sourceL, d)
        K: (n_context, sourceL, d)
        return (n_context, sourceL, sourceL)
        """
        batch_size, sourceL = K.size(0), K.size(1)
        K = torch.transpose(K, 1, 2).contiguous()
        attn = torch.bmm(Q, K)

        attn = attn.view(batch_size * sourceL, sourceL)
        attn = nn.Softmax(dim=1)(attn * xlambda)
        attn = attn.view(batch_size, sourceL, -1)
        return attn


def random_point_dropout(pc, max_dropout_ratio=0.875):
    ''' batch_pc: Nx3 '''

    dropout_ratio = np.random.random() * max_dropout_ratio # 0~0.875
    drop_idx = np.where(np.random.random((pc.shape[0])) <= dropout_ratio)[0]

    if len(drop_idx) > 0:
        pc[drop_idx,:] = pc[0,:]  # set to the first point

    return pc

def random_scale_point_cloud(batch_data, scale_low=0.8, scale_high=1.25):
    """ Randomly scale the point cloud. Scale is per point cloud.
        Input:
            Nx3 array, original batch of point clouds
        Return:
            Nx3 array, scaled batch of point clouds
    """
    scales = np.random.uniform(scale_low, scale_high)
    batch_data *= scales

    return batch_data

def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          Nx3 array, original batch of point clouds
        Return:
          Nx3 array, jittered batch of point clouds
    """
    N, C = batch_data.shape
    assert(clip > 0)

    jittered_data = np.clip(sigma * np.random.randn(N, C), -1*clip, clip).astype(float)
    batch_data += jittered_data

    return batch_data

def shuffle_points(batch_data):
    """ Shuffle orders of points in each point cloud -- changes FPS behavior.
        Use the same shuffling idx for the entire batch.
        Input:
            NxC array
        Output:
            NxC array
    """
    idx = np.arange(batch_data.shape[0])
    np.random.shuffle(idx)
    return batch_data[idx,:]

def rotate_point_cloud_z(pc):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          Nx3 array, original point clouds
        Return:
          Nx3 array, rotated point clouds
    """
    rotation_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, sinval, 0],
                                [-sinval, cosval, 0],
                                [0, 0, 1]])
    rotated_data = np.dot(pc, rotation_matrix)
    return rotated_data

def show_point_clouds(pts, out):
    fout = open(out, 'w')
    for i in range(pts.shape[0]):
        fout.write('v %f %f %f %d %d %d\n' % (
            pts[i, 0], pts[i, 1], pts[i, 2], 0, 255, 255))
    fout.close()


def tensor2points(tensor, offset=(-80., -80., -5.), voxel_size=(.05, .05, .1)):
    indices = tensor.float()
    voxel_size = torch.Tensor(voxel_size).to(indices.device)
    indices[:, :3] = indices[:, :3] * voxel_size + offset + .5 * voxel_size
    return indices