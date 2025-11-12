from torch.autograd import Function
import torch
from modules.functional.backend import _backend

__all__ = ['trilinear_devoxelize']


# class TrilinearDevoxelization(Function):
#     @staticmethod
#     def forward(ctx, features, coords, resolution, is_training=True):
#         """
#         :param ctx:
#         :param coords: the coordinates of points, FloatTensor[B, 3, N]
#         :param features: FloatTensor[B, C, R, R, R]
#         :param resolution: int, the voxel resolution
#         :param is_training: bool, training mode
#         :return:
#             FloatTensor[B, C, N]
#         """
#         B, C = features.shape[:2]
#         features = features.contiguous().view(B, C, -1)
#         coords = coords.contiguous()
#         outs, inds, wgts = _backend.trilinear_devoxelize_forward(resolution, is_training, coords, features)
#         if is_training:
#             ctx.save_for_backward(inds, wgts)
#             ctx.r = resolution
#         return outs

#     @staticmethod
#     def backward(ctx, grad_output):
#         """
#         :param ctx: 
#         :param grad_output: gradient of outputs, FloatTensor[B, C, N]
#         :return:
#             gradient of inputs, FloatTensor[B, C, R, R, R]
#         """
#         inds, wgts = ctx.saved_tensors
#         grad_inputs = _backend.trilinear_devoxelize_backward(grad_output.contiguous(), inds, wgts, ctx.r)
#         return grad_inputs.view(grad_output.size(0), grad_output.size(1), ctx.r, ctx.r, ctx.r), None, None, None



class TrilinearDevoxelization(Function):
    @staticmethod
    def forward(ctx, features, coords, resolution, is_training=True, tooth_mask=None):
        """
        :param features:  FloatTensor[B, C, R, R, R]
        :param coords:    FloatTensor[B, 3, N]
        :param resolution: int, voxel resolution
        :param is_training: bool
        :param tooth_mask: ByteTensor/BoolTensor (B,) => 1 if tooth present, 0 if missing
        :return: FloatTensor[B, C, N]
        """
        # If tooth_mask is None, treat all as valid
        if tooth_mask is None:
            tooth_mask = features.new_ones(features.size(0), dtype=torch.bool)

        B, C, R, _, _ = features.shape
        N = coords.shape[2]

        # 1. Gather only the valid (non-missing) indices
        valid_inds = tooth_mask.nonzero(as_tuple=False).view(-1)  # shape: (B_valid,)
        B_valid = valid_inds.numel()

        # If everything is missing, return zeros (no interpolation done)
        if B_valid == 0:
            outs = features.new_zeros((B, C, N))
            if is_training:
                # Save dummy tensors to keep the autograd happy
                ctx.save_for_backward(
                    torch.empty(0, dtype=torch.long, device=features.device),
                    torch.empty(0, dtype=features.dtype, device=features.device),
                )
                ctx.r = resolution
            return outs

        # 2. Subset for valid teeth
        features_valid = features[valid_inds]  # (B_valid, C, R, R, R)
        coords_valid   = coords[valid_inds]    # (B_valid, 3, N)

        # Reshape for the kernel
        features_valid = features_valid.contiguous().view(B_valid, C, -1)
        coords_valid   = coords_valid.contiguous()

        # 3. Call the native kernel only on valid subset
        outs_valid, inds_valid, wgts_valid = _backend.trilinear_devoxelize_forward(
            resolution, is_training, coords_valid, features_valid
        )
        # outs_valid: (B_valid, C, N)

        # 4. Allocate full output & scatter back into B-dim shape
        outs = features.new_zeros((B, C, N))
        outs[valid_inds] = outs_valid

        # 5. Save for backward
        if is_training:
            ctx.save_for_backward(inds_valid, wgts_valid, valid_inds)
            ctx.r = resolution
            ctx.B = B
            ctx.C = C
            ctx.N = N

        return outs

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param grad_output: (B, C, N)
        :return: gradient wrt features => (B, C, R, R, R)
        """
        inds_valid, wgts_valid, valid_inds = ctx.saved_tensors
        resolution = ctx.r
        B, C, N = ctx.B, ctx.C, ctx.N

        # 1. Gather only the valid grads
        grad_output_valid = grad_output[valid_inds]  # (B_valid, C, N)

        # 2. Call native backward on the valid subset
        grad_inputs_valid = _backend.trilinear_devoxelize_backward(
            grad_output_valid.contiguous(), inds_valid, wgts_valid, resolution
        )
        # shape: (B_valid, C, R*R*R)

        grad_inputs_valid = grad_inputs_valid.view(-1, C, resolution, resolution, resolution)

        # 3. Allocate full grad
        grad_inputs = grad_output.new_zeros((B, C, resolution, resolution, resolution))
        grad_inputs[valid_inds] = grad_inputs_valid

        # We return (grad_features, None, None, None, None)
        # or with the correct number of Nones for the extra arguments
        return grad_inputs, None, None, None, None
    


trilinear_devoxelize = TrilinearDevoxelization.apply
