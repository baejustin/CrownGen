import torch as th
import torch.nn as nn


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)

class CheckpointFunction(th.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        with th.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors
    
def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

def normalization(channels):
    """
    Make a standard normalization layer.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(32, channels)


def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.

    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)




class RPENet(nn.Module):
    def __init__(self, channels, num_heads, time_embed_dim):
        super().__init__()
        self.embed_distances = nn.Linear(3, channels)
        self.embed_diffusion_time = nn.Linear(time_embed_dim, channels)
        self.silu = nn.SiLU()
        self.out = nn.Linear(channels, channels)
        self.out.weight.data *= 0.
        self.out.bias.data *= 0.
        self.channels = channels
        self.num_heads = num_heads

    def forward(self, femb, relative_distances):
        distance_embs = th.stack(
            [th.log(1+(relative_distances).clamp(min=0)),
             th.log(1+(-relative_distances).clamp(min=0)),
             (relative_distances == 0).float()],
            dim=-1
        )  # BxTxTx3
        B, T, _ = relative_distances.shape
        C = self.channels

        emb = self.embed_diffusion_time(femb).view(B, T, 1, C) \
            + self.embed_distances(distance_embs)  # B x T x T x C
        return self.out(self.silu(emb)).view(*relative_distances.shape, self.num_heads, self.channels//self.num_heads)


class RPE(nn.Module):
    # Based on https://github.com/microsoft/Cream/blob/6fb89a2f93d6d97d2c7df51d600fe8be37ff0db4/iRPE/DeiT-with-iRPE/rpe_vision_transformer.py
    def __init__(self, channels, num_heads, time_embed_dim, use_rpe_net=False):
        """ This module handles the relative positional encoding.
        Args:
            channels (int): Number of input channels.
            num_heads (int): Number of attention heads.
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // self.num_heads
        self.use_rpe_net = use_rpe_net
        if use_rpe_net:
            self.rpe_net = RPENet(channels, num_heads, time_embed_dim)
        else:
            self.lookup_table_weight = nn.Parameter(
                th.zeros(2 * self.beta + 1,
                         self.num_heads,
                         self.head_dim))

    def get_R(self, pairwise_distances, femb):
        if self.use_rpe_net:
            return self.rpe_net(femb, pairwise_distances)
        else:
            return self.lookup_table_weight[pairwise_distances]  # BxTxTxHx(C/H)

    def forward(self, x, pairwise_distances, femb, mode):
        if mode == "qk":
            return self.forward_qk(x, pairwise_distances, femb)
        elif mode == "v":
            return self.forward_v(x, pairwise_distances, femb)
        else:
            raise ValueError(f"Unexpected RPE attention mode: {mode}")

    def forward_qk(self, qk, pairwise_distances, femb):
        # qv is either of q or k and has shape BxDxHxTx(C/H)
        # Output shape should be # BxDxHxTxT
        R = self.get_R(pairwise_distances, femb)
        return th.einsum(  # See Eq. 16 in https://arxiv.org/pdf/2107.14222.pdf
            "bdhtf,btshf->bdhts", qk, R  # BxDxHxTxT
        )

    def forward_v(self, attn, pairwise_distances, femb):
        # attn has shape BxDxHxTxT
        # Output shape should be # BxDxHxYx(C/H)
        R = self.get_R(pairwise_distances, femb)
        th.einsum("bdhts,btshf->bdhtf", attn, R)
        return th.einsum(  # See Eq. 16ish in https://arxiv.org/pdf/2107.14222.pdf
            "bdhts,btshf->bdhtf", attn, R  # BxDxHxTxT
        )

    def forward_safe_qk(self, x, pairwise_distances, femb):
        R = self.get_R(pairwise_distances, femb)
        B, T, _, H, F = R.shape
        D = x.shape[1]
        res = x.new_zeros(B, D, H, T, T) # attn shape
        for b in range(B):
            for d in range(D):
                for h in range(H):
                    for i in range(T):
                        for j in range(T):
                            res[b, d, h, i, j] = x[b, d, h, i].dot(R[b, i, j, h])
        return res
    

class RPEAttention(nn.Module):
    # Based on https://github.com/microsoft/Cream/blob/6fb89a2f93d6d97d2c7df51d600fe8be37ff0db4/iRPE/DeiT-with-iRPE/rpe_vision_transformer.py#L42
    def __init__(self, channels, num_heads, use_checkpoint=False,
                 time_embed_dim=None, use_rpe_net=None,
                 use_rpe_q=True, use_rpe_k=True, use_rpe_v=True,
                 ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = channels // num_heads
        self.scale = head_dim ** -0.5
        self.use_checkpoint = use_checkpoint

        self.qkv = nn.Linear(channels, channels * 3)
        self.proj_out = zero_module(nn.Linear(channels, channels))
        self.norm = normalization(channels)

        if use_rpe_q or use_rpe_k or use_rpe_v:
            assert use_rpe_net is not None
        def make_rpe_func():
            return RPE(
                channels=channels, num_heads=num_heads,
                time_embed_dim=time_embed_dim, use_rpe_net=use_rpe_net,
            )
        self.rpe_q = make_rpe_func() if use_rpe_q else None
        self.rpe_k = make_rpe_func() if use_rpe_k else None
        self.rpe_v = make_rpe_func() if use_rpe_v else None

    def forward(self, x, femb, dentition_fdi_indices, attn_mask=None):
        out = checkpoint(self._forward, (x, femb, dentition_fdi_indices, attn_mask), self.parameters(), self.use_checkpoint)
        return out

    def _forward(self, x, femb, dentition_fdi_indices, attn_mask):
        B, D, C, T = x.shape
        x = x.reshape(B*D, C, T)
        x = self.norm(x)
        x = x.view(B, D, C, T)
        x = th.einsum("BDCT -> BDTC", x)  
        qkv = self.qkv(x).reshape(B, D, T, 3, self.num_heads, C // self.num_heads)
        qkv = th.einsum("BDTtHF -> tBDHTF", qkv)
        q, k, v = qkv[0], qkv[1], qkv[2]   
        q *= self.scale
        attn = (q @ k.transpose(-2, -1)) 
        if self.rpe_q is not None or self.rpe_k is not None or self.rpe_v is not None:
            pairwise_distances = (dentition_fdi_indices.unsqueeze(-1) - dentition_fdi_indices.unsqueeze(-2)) # BxTxT

        if self.rpe_k is not None:
            attn += self.rpe_k(q, pairwise_distances, femb=femb, mode="qk")

        if self.rpe_q is not None:
            attn += self.rpe_q(k * self.scale, pairwise_distances, femb=femb, mode="qk").transpose(-1, -2)

        def softmax(w, attn_mask):
            if attn_mask is not None:
                allowed_interactions = attn_mask.view(B, 1, T) * attn_mask.view(B, T, 1)
                allowed_interactions += (1-attn_mask.view(B, 1, T)) * (1-attn_mask.view(B, T, 1))
                inf_mask = (1-allowed_interactions)
                inf_mask[inf_mask == 1] = th.inf
                w = w - inf_mask.view(B, 1, 1, T, T)  
            return th.softmax(w.float(), dim=-1).type(w.dtype)

        attn = softmax(attn, attn_mask)
        out = attn @ v

        if self.rpe_v is not None:
            out += self.rpe_v(attn, pairwise_distances, femb=femb, mode="v")
        out = th.einsum("BDHTF -> BDTHF", out).reshape(B, D, T, C)
        out = self.proj_out(out)
        x = x + out
        x = th.einsum("BDTC -> BDCT", x)
        return x
    

class IntertoothAttentionBlock(nn.Module):

    def __init__(self, channels, num_heads, use_rpe_net, time_embed_dim=None, use_checkpoint=False):
        super().__init__()

        # unused
        self.spatial_attention = RPEAttention(
            channels=channels, num_heads=num_heads, use_checkpoint=use_checkpoint,
            use_rpe_q=False, use_rpe_k=False, use_rpe_v=False,
        )

        self.intertooth_attention = RPEAttention(
            channels=channels, num_heads=num_heads, use_checkpoint=use_checkpoint,
            time_embed_dim=time_embed_dim, use_rpe_net=use_rpe_net,
        )

    def forward(self, x, attn_mask, femb, T, dentition_fdi_indices=None):
        BT, C, P = x.shape
        B = BT//T
        # reshape to have T in the last dimension becuase that's what we attend over. This is the number of tooth in the dentition.
        x = x.view(B, T, C, P).permute(0, 3, 2, 1)  # B, P, C, T
        x = self.intertooth_attention(x,
                                    femb,
                                    dentition_fdi_indices,
                                    attn_mask=attn_mask.flatten(start_dim=2).squeeze(dim=2))

        x = x.permute(0,3,2,1).reshape(BT, C, P)
        return x
