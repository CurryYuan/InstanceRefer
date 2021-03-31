from torch.autograd import Function

import torchsparse_cuda
from torchsparse.nn.functional.hash import *

__all__ = ['spvoxelize']


class VoxelizeGPU(Function):
    @staticmethod
    def forward(ctx, feat, idx, cnt):
        out = torchsparse_cuda.insertion_forward(feat.float().contiguous(),
                                                 idx.int().contiguous(), cnt)
        ctx.for_backwards = (idx.int().contiguous(), cnt, feat.shape[0])
        return out

    @staticmethod
    def backward(ctx, top_grad):
        idx, cnt, N = ctx.for_backwards
        bottom_grad = torchsparse_cuda.insertion_backward(
            top_grad.float().contiguous(), idx, cnt, N)
        return bottom_grad, None, None


voxelize_gpu = VoxelizeGPU.apply


def spvoxelize(feat, idx, cnt):
    return voxelize_gpu(feat, idx, cnt)
