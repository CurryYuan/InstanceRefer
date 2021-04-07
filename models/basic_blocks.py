import torch
import math
import torch.nn as nn
import torchsparse.nn as spnn

from torchsparse import SparseTensor
from torch_geometric.nn import MessagePassing, knn_graph, knn


class BasicConvolutionBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1, transpose=False):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc,
                        outc,
                        kernel_size=ks,
                        dilation=dilation,
                        stride=stride,
                        transpose=transpose),
            spnn.BatchNorm(outc),
            spnn.ReLU(True))

    def forward(self, x):
        out = self.net(x)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc,
                        outc,
                        kernel_size=ks,
                        dilation=dilation,
                        stride=stride),
            spnn.BatchNorm(outc),
            spnn.ReLU(True),
            spnn.Conv3d(outc,
                        outc,
                        kernel_size=ks,
                        dilation=dilation,
                        stride=1),
            spnn.BatchNorm(outc))

        self.downsample = nn.Sequential() if (inc == outc and stride == 1) else \
            nn.Sequential(
                spnn.Conv3d(inc, outc, kernel_size=1, dilation=1, stride=stride),
                spnn.BatchNorm(outc)
            )

        self.relu = spnn.ReLU(True)

    def forward(self, x):
        out = self.relu(self.net(x) + self.downsample(x))
        return out


class SparseEncoder(nn.Module):
    def __init__(self, input_dim, out):
        super().__init__()

        self.stem = nn.Sequential(
            BasicConvolutionBlock(input_dim, 32, 3)
        )

        self.stage1 = nn.Sequential(
            BasicConvolutionBlock(32, 64, ks=2, stride=2),
            ResidualBlock(64, 64, 3),
        )

        self.stage2 = nn.Sequential(
            BasicConvolutionBlock(64, 128, ks=2, stride=2),
            ResidualBlock(128, 128, 3),
        )

        self.stage3 = nn.Sequential(
            BasicConvolutionBlock(128, 128, ks=2, stride=2),
            ResidualBlock(128, out, 3),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)

        return x


class BEVEncoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.stem = nn.Sequential(
            BasicConvolutionBlock(input_dim, 32, 3)
        )

        self.stage1 = nn.Sequential(
            BasicConvolutionBlock(32, 64, ks=2, stride=2),
            ResidualBlock(64, 64, 3),
        )

        self.stage2 = nn.Sequential(
            BasicConvolutionBlock(64, 128, ks=2, stride=2),
            ResidualBlock(128, 128, 3),
        )

        self.stage3 = nn.Sequential(
            BasicConvolutionBlock(128, 128, ks=2, stride=2),
            ResidualBlock(128, 128, 3),
        )

        self.stage4 = nn.Sequential(
            BasicConvolutionBlock(128, 128, ks=2, stride=2),
            ResidualBlock(128, 128, 3),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        return x


class DynamicEdgeConv(MessagePassing):
    def __init__(self, F_in, F_out, args):
        super(DynamicEdgeConv, self).__init__(aggr='max')
        self.args = args
        self.k = args.k
        self.mlp = nn.Sequential(
            nn.Linear(3 * F_in, F_out),
            nn.ReLU(),
            nn.Linear(F_out, F_out)
        )
        self.weight = nn.Sequential(
            nn.Linear(3 + self.args.num_classes * 2, 64),
            nn.ReLU(),
            nn.Linear(64, F_in)
        )

    def forward(self, support_xyz, batch_index, filtered_index, features):
        # knn
        query_xyz = torch.index_select(support_xyz, 0, filtered_index)
        query_batch_index = torch.index_select(batch_index, 0, filtered_index)
        query_features = torch.index_select(features, 0, filtered_index)

        row, col = knn(support_xyz, query_xyz, self.k, batch_index, query_batch_index)
        edge_index = torch.stack([col, row], dim=0)

        # x has shape [N, F_in]
        # edge_index has shape [2, E]
        return self.propagate(edge_index, x=(features, query_features),
                              pos=(support_xyz, query_xyz))  # shape [N, F_out]

    def message(self, x_i, x_j, pos_i, pos_j):
        # x_i has shape [E, F_in]
        # x_j has shape [E, F_in]
        edge_weights = self.weight(
            torch.cat([pos_j - pos_i, x_i[:, -self.args.num_classes:], x_j[:, -self.args.num_classes:]], -1))
        edge_features = torch.cat([x_i, edge_weights, x_j], dim=1)  # shape [E, 3 * F_in]

        return self.mlp(edge_features)  # shape [E, F_out]


def masked_softmax(vector: torch.Tensor,
                   mask: torch.Tensor,
                   dim: int = -1,
                   memory_efficient: bool = False,
                   mask_fill_value: float = -1e32) -> torch.Tensor:
    """
    ``torch.nn.functional.softmax(vector)`` does not work if some elements of ``vector`` should be
    masked.  This performs a softmax on just the non-masked portions of ``vector``.  Passing
    ``None`` in for the mask is also acceptable; you'll just get a regular softmax.
    ``vector`` can have an arbitrary number of dimensions; the only requirement is that ``mask`` is
    broadcastable to ``vector's`` shape.  If ``mask`` has fewer dimensions than ``vector``, we will
    unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
    do it yourself before passing the mask into this function.
    If ``memory_efficient`` is set to true, we will simply use a very large negative number for those
    masked positions so that the probabilities of those positions would be approximately 0.
    This is not accurate in math, but works for most cases and consumes less memory.
    In the case that the input vector is completely masked and ``memory_efficient`` is false, this function
    returns an array of ``0.0``. This behavior may cause ``NaN`` if this is used as the last layer of
    a model that uses categorical cross-entropy loss. Instead, if ``memory_efficient`` is true, this function
    will treat every element as equal, and do softmax over equal numbers.
    """
    if mask is None:
        result = torch.nn.functional.softmax(vector, dim=dim)
    else:
        mask = mask.bool()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        if not memory_efficient:
            # To limit numerical errors from large vector elements outside the mask, we zero these out.
            result = torch.nn.functional.softmax(vector * mask, dim=dim)
            result = result * mask
            result = result / (result.sum(dim=dim, keepdim=True) + 1e-13)
        else:
            masked_vector = vector.masked_fill(mask.logical_not(), mask_fill_value)
            result = torch.nn.functional.softmax(masked_vector, dim=dim)

    return result


def spcrop(inputs, loc_min, loc_max):
    features = inputs.F
    coords = inputs.C
    cur_stride = inputs.s

    valid_flag = ((coords[:, :3] >= loc_min) & (coords[:, :3] < loc_max)).all(-1)
    output_coords = coords[valid_flag]
    output_features = features[valid_flag]
    return SparseTensor(output_features, output_coords, cur_stride)


class SparseCrop(nn.Module):
    def __init__(self, loc_min, loc_max):
        super().__init__()
        self.loc_min = loc_min
        self.loc_max = loc_max

    def forward(self, inputs):
        return spcrop(inputs, self.loc_min, self.loc_max)


class ToDenseBEVConvolution(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 shape,
                 offset: list = [0, 0, 0],
                 z_dim: int = 1,
                 use_bias: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.offset = torch.cuda.IntTensor([list(offset) + [0]])
        self.z_dim = z_dim
        self.n_kernels = int(shape[self.z_dim])
        self.bev_dims = [i for i in range(3) if i != self.z_dim]
        self.bev_shape = shape[self.bev_dims]
        self.kernel = nn.Parameter(torch.zeros(self.n_kernels, in_channels, out_channels))
        self.bias = nn.Parameter(torch.zeros(1, out_channels)) if use_bias else 0
        self.init_weight()

    def __repr__(self):
        return 'ToDenseBEVConvolution(in_channels=%d, out_channels=%d, n_kernels=%d)' % (
            self.in_channels,
            self.out_channels,
            self.n_kernels
        )

    def init_weight(self):
        std = 1. / math.sqrt(self.in_channels)
        self.kernel.data.uniform_(-std, std)

    def forward(self, inputs):
        features = inputs.F
        coords = inputs.C
        cur_stride = inputs.s

        kernels = torch.index_select(self.kernel, 0, coords[:, self.z_dim].long() // cur_stride)
        sparse_features = (features.unsqueeze(-1) * kernels).sum(1) + self.bias
        sparse_coords = (coords - self.offset).t()[[3] + self.bev_dims].long()
        sparse_coords[1:] = sparse_coords[1:] // cur_stride
        batch_size = sparse_coords[0].max().item() + 1
        sparse_coords = sparse_coords[0] * int(self.bev_shape.prod()) + sparse_coords[1] * int(self.bev_shape[1]) + \
                        sparse_coords[2]
        bev = torch.cuda.sparse.FloatTensor(
            sparse_coords.unsqueeze(0),
            sparse_features,
            torch.Size([batch_size * int(self.bev_shape.prod()), sparse_features.size(-1)]),
        ).to_dense()

        return bev.view(batch_size, *self.bev_shape, -1).permute(0, 3, 1, 2).contiguous()  # To BCHW


class PositionalEncoding(nn.Module):
    """Implement the PE function."""

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # (max_len, 1, C)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = torch.autograd.Variable(self.pe[: x.size(0)], requires_grad=False)
        return self.dropout(x)


def length_to_mask(length, max_len=None, dtype=None):
    """length: B.
    return B x max_len.
    If max_len is None, then max of length will be used.
    """
    assert len(length.shape) == 1, "Length shape should be 1 dimensional."
    max_len = max_len or length.max().item()
    mask = torch.arange(max_len, device=length.device, dtype=length.dtype).expand(
        len(length), max_len
    ) < length.unsqueeze(1)
    if dtype is not None:
        mask = torch.as_tensor(mask, dtype=dtype, device=length.device)
    return mask
