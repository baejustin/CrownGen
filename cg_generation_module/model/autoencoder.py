import functools
import torch.nn as nn
import torch
import numpy as np
from modules import (
    SharedMLP, 
    PVConv, 
    PointNetSAModule, 
    PointNetAModule, 
    PointNetFPModule, 
    Swish, 
    FactorizedAttentionBlock)
from model import Transformer, LayerNorm

def _linear_gn_relu(in_channels, out_channels):
    return nn.Sequential(
        nn.Linear(in_channels, out_channels),
        nn.GroupNorm(8, out_channels),
        Swish()
    )

def create_mlp_components(in_channels, out_channels, classifier=False, dim=2, width_multiplier=1):

    width_mult = width_multiplier

    if dim == 1:
        block = _linear_gn_relu
    else:
        block = SharedMLP

    if not isinstance(out_channels, (list, tuple)):
        out_channels = [out_channels]

    if len(out_channels) == 0 or (len(out_channels) == 1 and out_channels[0] is None):
        return nn.Sequential(), in_channels, in_channels

    layers = []
    for oc in out_channels[:-1]:
        if oc < 1:
            layers.append(nn.Dropout(oc))
        else:
            oc = int(width_mult * oc)
            layers.append(block(in_channels, oc))
            in_channels = oc

    if dim == 1:
        if classifier:
            layers.append(nn.Linear(in_channels, out_channels[-1]))
        else:
            layers.append(_linear_gn_relu(in_channels, int(width_mult * out_channels[-1])))
    else:
        if classifier:
            layers.append(nn.Conv1d(in_channels, out_channels[-1], 1))
        else:
            layers.append(SharedMLP(in_channels, int(width_mult * out_channels[-1])))

    final_channels = out_channels[-1] if classifier else int(width_mult * out_channels[-1])
    return layers, final_channels


def create_pointnet_components(blocks, in_channels, embed_dim, with_se=False, normalize=True, eps=0,
                               width_multiplier=1, voxel_resolution_multiplier=1):
    width_mult = width_multiplier
    voxel_res_mult = voxel_resolution_multiplier

    layers = []
    concat_channels = 0
    block_idx = 0

    for stage_idx, (out_channels, num_blocks, voxel_resolution) in enumerate(blocks):
        out_channels = int(width_mult * out_channels)

        for block_in_stage in range(num_blocks):
            attention = stage_idx % 2 == 0 and stage_idx > 0 and block_in_stage == 0

            if voxel_resolution is None:
                block = SharedMLP
            else:
                block = functools.partial(
                    PVConv,
                    kernel_size=3,
                    resolution=int(voxel_res_mult * voxel_resolution),
                    attention=attention,
                    with_se=with_se,
                    normalize=normalize,
                    eps=eps
                )

            if block_idx == 0:
                layers.append(block(in_channels, out_channels))
            else:
                layers.append(block(in_channels + embed_dim, out_channels))

            in_channels = out_channels
            concat_channels += out_channels
            block_idx += 1

    return layers, in_channels, concat_channels


def create_pointnet2_sa_components(sa_blocks, extra_feature_channels, embed_dim=64, use_att=False,
                                   dropout=0.1, with_se=False, normalize=True, eps=0,
                                   width_multiplier=1, voxel_resolution_multiplier=1):

    width_mult = width_multiplier
    voxel_res_mult = voxel_resolution_multiplier
    in_channels = extra_feature_channels + 3

    sa_layers = []
    sa_in_channels = []
    stage_idx = 0

    for conv_configs, sa_configs in sa_blocks:
        conv_block_idx = 0
        sa_in_channels.append(in_channels)
        stage_blocks = []

        if conv_configs is not None:
            out_channels, num_blocks, voxel_resolution = conv_configs
            out_channels = int(width_mult * out_channels)

            for block_in_stage in range(num_blocks):

                attention = (stage_idx + 1) % 2 == 0 and stage_idx > 0 and use_att and block_in_stage == 0

                if voxel_resolution is None:
                    block = SharedMLP
                else:
                    block = functools.partial(
                        PVConv,
                        kernel_size=3,
                        resolution=int(voxel_res_mult * voxel_resolution),
                        attention=attention,
                        dropout=dropout,
                        with_se=with_se and not attention,
                        with_se_relu=True,
                        normalize=normalize,
                        eps=eps
                    )

                if stage_idx == 0:
                    stage_blocks.append(block(in_channels, out_channels))
                elif conv_block_idx == 0:
                    stage_blocks.append(block(in_channels + embed_dim, out_channels))

                in_channels = out_channels
                conv_block_idx += 1

            extra_feature_channels = in_channels


        num_centers, radius, num_neighbors, out_channels = sa_configs

        scaled_out_channels = []
        for oc in out_channels:
            if isinstance(oc, (list, tuple)):
                scaled_out_channels.append([int(width_mult * _oc) for _oc in oc])
            else:
                scaled_out_channels.append(int(width_mult * oc))
        out_channels = scaled_out_channels

        # Select SA module type
        if num_centers is None:
            sa_block = PointNetAModule
        else:
            sa_block = functools.partial(
                PointNetSAModule,
                num_centers=num_centers,
                radius=radius,
                num_neighbors=num_neighbors
            )

        sa_input_channels = extra_feature_channels + (embed_dim if conv_block_idx == 0 else 0)
        stage_blocks.append(
            sa_block(
                in_channels=sa_input_channels,
                out_channels=out_channels,
                include_coordinates=True
            )
        )

        stage_idx += 1
        in_channels = extra_feature_channels = stage_blocks[-1].out_channels

        if len(stage_blocks) == 1:
            sa_layers.append(stage_blocks[0])
        else:
            sa_layers.append(nn.Sequential(*stage_blocks))

    num_centers = 1 if num_centers is None else num_centers
    return sa_layers, sa_in_channels, in_channels, num_centers





def create_pointnet2_fp_modules(fp_blocks, in_channels, sa_in_channels, embed_dim=64, use_att=False,
                                dropout=0.1, with_se=False, normalize=True, eps=0,
                                width_multiplier=1, voxel_resolution_multiplier=1):

    width_mult = width_multiplier
    voxel_res_mult = voxel_resolution_multiplier

    fp_layers = []
    stage_idx = 0

    for fp_idx, (fp_configs, conv_configs) in enumerate(fp_blocks):
        fp_blocks = []
        out_channels = tuple(int(width_mult * oc) for oc in fp_configs)
        fp_blocks.append(
            PointNetFPModule(in_channels=in_channels + sa_in_channels[-1 - fp_idx] + embed_dim,
                             out_channels=out_channels)
        )
        in_channels = out_channels[-1]

        if conv_configs is not None:
            out_channels, num_blocks, voxel_resolution = conv_configs
            out_channels = int(width_mult * out_channels)
            for block_in_stage in range(num_blocks):
                attention = stage_idx % 2 == 0 and stage_idx < len(fp_blocks) - 1 and use_att and block_in_stage == 0

                if voxel_resolution is None:
                    block = SharedMLP
                else:
                    block = functools.partial(
                        PVConv,
                        kernel_size=3,
                        resolution=int(voxel_res_mult * voxel_resolution),
                        attention=attention,
                        dropout=dropout,
                        with_se=with_se and not attention,
                        with_se_relu=True,
                        normalize=normalize,
                        eps=eps
                    )
                
                fp_blocks.append(block(in_channels, out_channels))
                in_channels = out_channels

        if len(fp_blocks) == 1:
            fp_layers.append(fp_blocks[0])
        else:
            fp_layers.append(nn.Sequential(*fp_blocks))

        stage_idx += 1

    return fp_layers, in_channels



class PVCNN2Base(nn.Module):

    def __init__(self, num_classes, embed_dim, use_att, dropout=0.1,
                 extra_feature_channels=3, width_multiplier=1, voxel_resolution_multiplier=1):
        super().__init__()
        assert extra_feature_channels >= 0
        self.embed_dim = embed_dim
        self.extra_feature_channels = extra_feature_channels
        self.in_channels = extra_feature_channels + 3


        self.fdi_embedding = nn.Embedding(num_embeddings=28, embedding_dim=8)


        sa_layers, sa_in_channels, channels_sa_features, _ = create_pointnet2_sa_components(
            sa_blocks=self.sa_blocks, extra_feature_channels=extra_feature_channels, with_se=True, embed_dim=embed_dim,
            use_att=use_att, dropout=dropout,
            width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier)


        self.sa_layers = nn.ModuleList(sa_layers)


        self.sa_att_dict = nn.ModuleDict({idx:FactorizedAttentionBlock(
            channels_sa_features//num_chan_div,
            use_checkpoint=False,
            num_heads = 4,
            use_rpe_net = True, 
            time_embed_dim = self.embed_dim  
        ) for idx,num_chan_div in {'1':8, '2':4, '3':2}.items()})



        self.global_att = None if not use_att else FactorizedAttentionBlock(
            channels_sa_features,
            use_checkpoint=False,
            num_heads = 4,
            use_rpe_net = True, 
            time_embed_dim = self.embed_dim  
        )

        self.fp_att_dict = nn.ModuleDict({idx:FactorizedAttentionBlock(
            channels_sa_features//num_chan_div,
            use_checkpoint=False,
            num_heads = 4,
            use_rpe_net = True, 
            time_embed_dim = self.embed_dim  
        ) for idx,num_chan_div in {'0':2, '1':2, '2':4}.items()})

        sa_in_channels[0] = extra_feature_channels
        fp_layers, channels_fp_features = create_pointnet2_fp_modules(
            fp_blocks=self.fp_blocks, in_channels=channels_sa_features, sa_in_channels=sa_in_channels,
            with_se=True, embed_dim=embed_dim,
            use_att=use_att, dropout=dropout,
            width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier)

        self.fp_layers = nn.ModuleList(fp_layers)


        layers, _ = create_mlp_components(in_channels=channels_fp_features, out_channels=[128, 0.5, num_classes],
                                          classifier=True, dim=2, width_multiplier=width_multiplier)


        self.classifier = nn.Sequential(*layers)

        self.embedf = nn.Sequential(nn.Linear(embed_dim, embed_dim),
                                    nn.LeakyReLU(0.1, inplace=True),
                                    nn.Linear(embed_dim, embed_dim))
        


        self.bound_embedding = nn.Linear(5, embed_dim)

        self.bound_transformer = Transformer(
            n_ctx=28,
            width = embed_dim,
            layers = 4,
            heads = 8
        )

        self.bound_final_ln = LayerNorm(embed_dim)


    def get_timestep_embedding(self, timesteps, device):

        half_dim = self.embed_dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.from_numpy(np.exp(np.arange(0, half_dim) * -emb)).float().to(device)

        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

        if self.embed_dim % 2 == 1:  # zero pad
            emb = nn.functional.pad(emb, (0, 1), "constant", 0)
        assert emb.shape == torch.Size([timesteps.shape[0], self.embed_dim])
        return emb


    def forward(self, xt, t, return_attn_weights, x0, l_mask, o_mask, bound):

        if return_attn_weights:
            extracted_intertooth_attn = {'sa':[], 'bottleneck':[], 'fp':[]}
        else:
            extracted_intertooth_attn = None

        B, nT, nD, nP = xt.shape
        t = t.view(B, 1).expand(B, nT).reshape(B*nT)

        dentition_fdi_indices = torch.arange(28, device=x0.device).unsqueeze(0).repeat(B,1)

        attn_mask = torch.ones_like(o_mask, device=x0.device)
        fdi_embeddings = self.fdi_embedding(dentition_fdi_indices) 

        bound_embedding = self.bound_embedding(bound) 

        bound_embedding_transformed = self.bound_transformer(bound_embedding)
 
        bound_embedding_transformed = self.bound_final_ln(bound_embedding_transformed).reshape(B*nT, self.embed_dim)
        
        temb_raw = self.embedf(self.get_timestep_embedding(t, xt.device))

        temb_raw = temb_raw + bound_embedding_transformed


        temb = temb_raw[:, :, None].expand(-1, -1, xt.shape[-1])

        obs_indicator = torch.ones_like(xt[:,:,:1,:]) * o_mask  

        fdi_embeddings = fdi_embeddings.unsqueeze(3).repeat(1, 1, 1, nP) 


        x = torch.cat([
            xt*l_mask + x0*o_mask,
            fdi_embeddings,
            obs_indicator
        ], dim=2)

        x = x.reshape(B*nT, nD+self.extra_feature_channels, nP)


        coords, features = x[:, :3, :].contiguous(), x

        coords_list, in_features_list = [], []

        for sa_idx, sa_blocks in enumerate(self.sa_layers):  
            str_sa_idx = str(sa_idx)
            in_features_list.append(features)
            coords_list.append(coords)

            if str_sa_idx in self.sa_att_dict:
                features = self.sa_att_dict[str_sa_idx](x=features, attn_mask = attn_mask, temb=temb_raw, T=nT, dentition_fdi_indices=dentition_fdi_indices, \
                    attn_weights_list = extracted_intertooth_attn['sa'] if return_attn_weights else None)


            if sa_idx == 0:
                features, coords, temb = sa_blocks((features, coords, temb))
            else:
                features, coords, temb = sa_blocks((torch.cat([features, temb], dim=1), coords, temb))


        in_features_list[0] = x[:, 3:, :].contiguous()


        if self.global_att is not None:
            features = self.global_att(x=features, attn_mask = attn_mask, temb=temb_raw, T=nT, dentition_fdi_indices=dentition_fdi_indices, \
                attn_weights_list = extracted_intertooth_attn['bottleneck'] if return_attn_weights else None)

  
        for fp_idx, fp_blocks in enumerate(self.fp_layers):  
            str_fp_idx = str(fp_idx)
            jump_coords = coords_list[-1 - fp_idx]
            fump_feats = in_features_list[-1 - fp_idx]

            features, coords, temb = fp_blocks((jump_coords, coords, torch.cat([features, temb], dim=1), fump_feats, temb))

            if str_fp_idx in self.fp_att_dict:
                features = self.fp_att_dict[str_fp_idx](x=features, attn_mask = attn_mask, temb=temb_raw, T=nT, dentition_fdi_indices=dentition_fdi_indices, \
                    attn_weights_list = extracted_intertooth_attn['fp'] if return_attn_weights else None)

        out = self.classifier(features)


        out = out.view(B, nT, nD, nP)


        return out, extracted_intertooth_attn

