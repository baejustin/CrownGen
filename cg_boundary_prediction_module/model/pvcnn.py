import functools
import torch
import torch.nn as nn
from modules import (
    SharedMLP,
    PVConv,
    PointNetSAModule,
    PointNetAModule,
    Swish,
    IntertoothAttentionBlock
)

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


class PVCNN2(nn.Module):

    def __init__(self, output_dim, sa_blocks, embed_dim, use_att, dropout=0.1,
                 extra_feature_channels=3, width_multiplier=1, voxel_resolution_multiplier=1):

        super().__init__()

        self.sa_blocks = sa_blocks
        assert extra_feature_channels >= 0
        self.embed_dim = embed_dim
        self.extra_feature_channels = extra_feature_channels
        self.in_channels = extra_feature_channels + 3
        self.output_dim = output_dim

        self.fdi_embedding = nn.Embedding(num_embeddings=28, embedding_dim=self.embed_dim)

        sa_layers, _, channels_sa_features, _ = create_pointnet2_sa_components(
            sa_blocks=self.sa_blocks,
            extra_feature_channels=extra_feature_channels,
            with_se=True,
            embed_dim=embed_dim,
            use_att=use_att,
            dropout=dropout,
            width_multiplier=width_multiplier,
            voxel_resolution_multiplier=voxel_resolution_multiplier
        )
        self.sa_layers = nn.ModuleList(sa_layers)

        self.sa_att_dict = nn.ModuleDict({
            idx: IntertoothAttentionBlock(
                int(channels_sa_features / num_chan_div),
                use_checkpoint=False,
                num_heads=4,
                use_rpe_net=True,
                time_embed_dim=self.embed_dim
            )
            for idx, num_chan_div in {'1': 4, '2': 2}.items()
        })

        self.global_att = None
        if use_att:
            self.global_att = IntertoothAttentionBlock(
                channels_sa_features,
                use_checkpoint=False,
                num_heads=4,
                use_rpe_net=True,
                time_embed_dim=self.embed_dim
            )

        final_pointnet_layer, _, _ = create_pointnet_components(
            [(channels_sa_features, 1, 8)],
            channels_sa_features,
            embed_dim,
            with_se=True,
            normalize=True,
            width_multiplier=width_multiplier,
            voxel_resolution_multiplier=voxel_resolution_multiplier
        )
        self.final_pointnet_layer = nn.ModuleList(final_pointnet_layer)

        # Boundary prediction head
        self.bound_fc = nn.Sequential(
            nn.Linear(channels_sa_features, channels_sa_features // 2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(channels_sa_features // 2, output_dim)
        )

    def forward(self, dentition_points, o_mask):

        B, nT, nD, nP = dentition_points.shape

        dentition_fdi_indices = torch.arange(28, device=dentition_points.device).unsqueeze(0).repeat(B, 1)
        attn_mask = o_mask

        fdi_embeddings = self.fdi_embedding(dentition_fdi_indices)
        femb_raw = fdi_embeddings.reshape(B * nT, self.embed_dim)
        femb = femb_raw[:, :, None].expand(-1, -1, nP)

        obs_indicator = torch.ones_like(dentition_points[:, :, :1, :]) * o_mask

        x = torch.cat([
            dentition_points * o_mask,
            obs_indicator
        ], dim=2)

        x = x.reshape(B * nT, nD + self.extra_feature_channels, nP)
        o_mask = o_mask.reshape(B * nT).squeeze()

        coords = x[:, :3, :].contiguous()
        features = x

        for sa_idx, sa_blocks in enumerate(self.sa_layers):
            str_sa_idx = str(sa_idx)

            if str_sa_idx in self.sa_att_dict:
                features = self.sa_att_dict[str_sa_idx](
                    x=features,
                    attn_mask=attn_mask,
                    femb=femb_raw,
                    T=nT,
                    dentition_fdi_indices=dentition_fdi_indices
                )

            if sa_idx == 0:
                features, coords, femb, o_mask = sa_blocks((features, coords, femb, o_mask))
            else:
                features, coords, femb, o_mask = sa_blocks(
                    (torch.cat([features, femb], dim=1), coords, femb, o_mask)
                )

        if self.global_att is not None:
            features = self.global_att(
                x=features,
                attn_mask=attn_mask,
                femb=femb_raw,
                T=nT,
                dentition_fdi_indices=dentition_fdi_indices
            )

        final_pointnet_out, _, _, o_mask = self.final_pointnet_layer[0](
            (features, coords, femb, o_mask)
        )
        final_pointnet_out = final_pointnet_out.mean(dim=2)

        out = self.bound_fc(final_pointnet_out).view(B, nT, self.output_dim)
        return out

        