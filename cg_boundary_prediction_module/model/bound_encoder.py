import torch
import torch.nn.functional as F
import random
import pytorch_lightning as pl
from model.pvcnn import PVCNN2

class BoundEncoder(pl.LightningModule):

    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.output_dim = cfg.model.output_dim
        self.dropout = cfg.model.dropout
        self.max_epochs = cfg.training.epochs
        self.lr = cfg.optimizer.lr
        self.end_lr = cfg.optimizer.end_lr
        self.beta1 = cfg.optimizer.beta1
        self.decay = cfg.optimizer.decay
        self.max_missing_teeth = cfg.data.max_missing_teeth

        sa_blocks = [
            ((16, 2, 32), (128, 0.1, 32, (32, 64))),
            ((32, 3, 16), (64, 0.2, 32, (64, 128))),
            ((64, 3, 8), (16, 0.4, 32, (128, 256)))
        ]

        self.model = PVCNN2(
            output_dim=self.output_dim,
            sa_blocks=sa_blocks,
            embed_dim=64,
            use_att=True,
            extra_feature_channels=1,
            width_multiplier=1.0,
            voxel_resolution_multiplier=1.0,
            dropout=self.dropout
        )

    def forward(self, dentition_points, exist_mask):
        return self.model(dentition_points, exist_mask)

    def _create_missing_mask(self, dentition_points):
        batch_size = dentition_points.shape[0]
        exist_mask = torch.ones_like(dentition_points[:, :, :1, :1])

        for b in range(batch_size):
            n_missing = random.randint(1, self.max_missing_teeth)
            missing_indices = torch.randperm(28)[:n_missing]
            exist_mask[b, missing_indices, 0, 0] = 0

        missing_mask = 1.0 - exist_mask
        return exist_mask, missing_mask

    def _compute_loss(self, pred_bound, gt_bound, missing_mask):
        loss_per_element = F.smooth_l1_loss(pred_bound, gt_bound, reduction='none')
        loss_masked = loss_per_element * missing_mask.unsqueeze(-1)
        loss = loss_masked.sum() / missing_mask.sum()
        return loss

    def training_step(self, batch, batch_idx):
        dentition_points = batch['dentition_points']  
        gt_bound = batch['bounds_cyl']  

        exist_mask, missing_mask = self._create_missing_mask(dentition_points)
        pred_bound = self(dentition_points, exist_mask)
        loss = self._compute_loss(pred_bound, gt_bound, missing_mask)

        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True
        )

        return loss

    def validation_step(self, batch, batch_idx):
        dentition_points = batch['dentition_points']  
        gt_bound = batch['bounds_cyl']  

        exist_mask, missing_mask = self._create_missing_mask(dentition_points)
        pred_bound = self(dentition_points, exist_mask)
        loss = self._compute_loss(pred_bound, gt_bound, missing_mask)

        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True
        )

        return {"val_loss": loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.decay,
            betas=(self.beta1, 0.999)
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.max_epochs,
            eta_min=self.end_lr,
            last_epoch=-1
        )
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
