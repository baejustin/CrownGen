import os
import random
import numpy as np
from omegaconf import OmegaConf

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.distributed as dist

from model.diffusion import Model, get_betas
from datasets.dentition_data import DentitionDataset, FDIS_ORDER
from datasets.dentition_aug import *
from torchvision import transforms

from util import (
    setup_logging, 
    setup_output_subdirs,
    export_dentition, 
    set_seed,
    remove_module_prefix
)

def get_dataset(cfg, mode='train'):
    aug_transforms = None
    if mode == 'train':
        aug_transforms = transforms.Compose([
            RandomMirror(p=0.5),
            RandomScale(scale=[0.95, 1.05], p=0.75),
            ShufflePoint(p=0.9)
        ])

    dataset = DentitionDataset(
        path=cfg.dataset.path,
        mode=mode,
        tooth_npoints=cfg.dataset.tooth_npoints,
        norm_mode=cfg.dataset.norm_mode,
        boundary_size_scale=cfg.dataset.boundary_size_scale,
        aug_transforms=aug_transforms
    )
    
    return dataset


def get_dataloader(cfg, dataset, local_rank, mode='train'):
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=cfg.ddp.world_size,
        rank=local_rank,
        shuffle=(mode == 'train')
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.data_loader.batch_size,
        sampler=sampler,
        num_workers=int(cfg.data_loader.num_workers),
        pin_memory=cfg.data_loader.pin_memory,
        persistent_workers=cfg.data_loader.persistent_workers,
        drop_last=False
    )
    
    return dataloader, sampler


def generate_val_samples(cfg, model, val_dataset, outf_syn, epoch, device):

    model.eval()

    sample_batch_size = 1
    val_sample_list = val_dataset.sample_patient(sample_batch_size)
    val_dentition_points = torch.stack([sample['dentition_points'] for sample in val_sample_list], 0).to(device)
    val_bound = torch.stack([sample['bounds_cyl'] for sample in val_sample_list], 0).to(device)
    val_dentition_ids = [sample['patient_id'] for sample in val_sample_list]

    val_shift = torch.stack([sample['shift'] for sample in val_sample_list], 0).numpy()
    val_scale = torch.stack([sample['scale'] for sample in val_sample_list], 0).numpy()

    latent_mask_val = torch.zeros_like(val_dentition_points[:,:,:1,:1]).to(device)

    for i in range(sample_batch_size):
        n_missing_val = random.randint(1, cfg.dataset.max_missing_teeth)
        missing_indices_val = torch.randperm(28)[:n_missing_val]
        latent_mask_val[i, missing_indices_val, 0, 0] = 1
    
    obs_mask_val = torch.ones_like(latent_mask_val) - latent_mask_val

    val_data_dict = {
        'x0': val_dentition_points,
        'l_mask': latent_mask_val,
        'o_mask': obs_mask_val,
        'bound': val_bound
    }


    with torch.no_grad():
        sample_out, _ = model.gen_samples(
            model_kwargs=val_data_dict,
            return_attn_weights=False
        )
        sample_out = sample_out.detach().cpu()
        obs_mask_val = obs_mask_val.detach().cpu()
        latent_mask_val = latent_mask_val.detach().cpu()

        val_dentition_points_np = val_dentition_points.detach().cpu().numpy()

        for i in range(sample_batch_size):
            missing_indices = np.where(latent_mask_val.numpy().squeeze())[0]
            missing_fdis = [FDIS_ORDER[i] for i in missing_indices]

            condition_points = val_dentition_points_np[i][obs_mask_val[i].view(-1).bool()]
            generated_points = sample_out[i][latent_mask_val[i].view(-1).bool()].numpy()
            ground_truth_points = val_dentition_points_np[i][latent_mask_val[i].view(-1).bool()]
            patient_id = val_dentition_ids[i]

            export_dentition('{}_{}_condition'.format(patient_id, missing_fdis), 
                                '%s/epoch_%03d' % (outf_syn, epoch), condition_points, shift=val_shift[i], scale=val_scale[i], combine_teeth=True)
            export_dentition('{}_{}_gen'.format(patient_id, missing_fdis), 
                                '%s/epoch_%03d' % (outf_syn, epoch), generated_points, shift=val_shift[i], scale=val_scale[i], combine_teeth=True)
            export_dentition('{}_{}_gt'.format(patient_id, missing_fdis), 
                                '%s/epoch_%03d' % (outf_syn, epoch), ground_truth_points, shift=val_shift[i], scale=val_scale[i], combine_teeth=True)

    model.train()


def train(local_rank, cfg, output_dir):
    
    set_seed(cfg.training.manual_seed)
    logger = setup_logging(output_dir)

    should_diag = local_rank == 0

    if should_diag:
        logger.info(OmegaConf.to_yaml(cfg))
        outf_syn, = setup_output_subdirs(output_dir, 'syn')

    dist.init_process_group(
        backend=cfg.ddp.backend,
        init_method='env://',
        world_size=cfg.ddp.world_size,
        rank=local_rank
    )

    cfg.data_loader.batch_size = int(cfg.data_loader.batch_size / cfg.ddp.ngpus_per_node)
    cfg.data_loader.num_workers = int(cfg.data_loader.num_workers / cfg.ddp.ngpus_per_node)

    train_dataset = get_dataset(cfg, mode='train')
    val_dataset = get_dataset(cfg, mode='val')

    dataloader, sampler = get_dataloader(cfg, train_dataset, local_rank, mode='train')

    betas = get_betas(
        cfg.diffusion.schedule_type,
        cfg.diffusion.beta_start,
        cfg.diffusion.beta_end,
        cfg.diffusion.time_num
    )
    model = Model(
        cfg,
        betas,
        cfg.diffusion.loss_type,
        cfg.diffusion.model_mean_type,
        cfg.diffusion.model_var_type,
        cfg.model.width_mult,
        cfg.model.vox_res_mult
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.decay,
        betas=(cfg.training.beta1, 0.999)
    )

    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)

    def _transform_(m):
        return nn.parallel.DistributedDataParallel(
            m, device_ids=[local_rank], output_device=local_rank)

    model = model.to(device)
    model.multi_gpu_wrapper(_transform_)

    if cfg.checkpoint.model_ckpt != '':
        ckpt = torch.load(cfg.checkpoint.model_ckpt, map_location=device)

        model.load_state_dict(ckpt['model_state'])

        logger.info('saved weight loaded!')

        optimizer.load_state_dict(ckpt['optimizer_state'])
        logger.info('optimizer status loaded')
        start_epoch = ckpt['epoch'] + 1

        del ckpt
    else:
        start_epoch = 0

    lr_decay_epochs = [(cfg.training.niter//4), 
                       (cfg.training.niter//4)*2, 
                       (cfg.training.niter//4)*3]

    for epoch in range(start_epoch, cfg.training.niter):
        sampler.set_epoch(epoch)
        logger.info('start epoch {}'.format(epoch))
        for i, data in enumerate(dataloader):
            
            dentition_points = data['dentition_points'].to(device)  # (b, 28, 3, 1024)
            B, nT, nD, nP = dentition_points.shape
            
            latent_mask = torch.zeros_like(dentition_points[:,:,:1,:1]).to(device)

            for b in range(B):
                n_missing = random.randint(1, cfg.dataset.max_missing_teeth)
                missing_indices = torch.randperm(28)[:n_missing]
                latent_mask[b, missing_indices, 0, 0] = 1

            obs_mask = torch.ones_like(latent_mask) - latent_mask
            noise_batch = torch.randn_like(dentition_points).to(device)

            bound = data['bounds_cyl'].to(device)
            
            data_dict = {
                'x0': dentition_points,
                'l_mask': latent_mask,
                'o_mask': obs_mask,
                'bound': bound
            }
            loss = model.get_loss_iter_teethmask(noise_batch, model_kwargs=data_dict)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % cfg.evaluation.diag_freq == 0 and should_diag:
                logger.info('[{:>3d}/{:>3d}][{:>3d}/{:>3d}]    loss: {:>10.4f},    '
                            .format(epoch, cfg.training.niter, i, len(dataloader), loss.item()))
                
      
        if (epoch+1) in lr_decay_epochs:
            for param_group in optimizer.param_groups:
                logger.info('old lr: {}'.format(param_group['lr']))
                param_group['lr'] = param_group['lr'] * cfg.training.lr_decay_factor
                logger.info('new lr: {}'.format(param_group['lr']))

        if (epoch + 1) % cfg.evaluation.viz_iter == 0:
            if should_diag:
                generate_val_samples(cfg, model, val_dataset, outf_syn, epoch, device)
            dist.barrier()

        if (epoch + 1) % cfg.evaluation.save_iter == 0:
            if should_diag:
                save_dict = {
                    'epoch': epoch,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict()
                }
                torch.save(save_dict, '%s/epoch_%d.pth' % (output_dir, epoch))
            
            dist.barrier()

    dist.destroy_process_group()


def inference(cfg):

    os.makedirs(cfg.inference.save_dir, exist_ok=True)

    test_dataset = DentitionDataset(
        path=cfg.inference.datapath,
        mode='test',
        tooth_npoints=cfg.dataset.tooth_npoints,
        norm_mode=cfg.dataset.norm_mode,
        boundary_size_scale=cfg.dataset.boundary_size_scale,
        aug_transforms=None
    )

    betas = get_betas(
        cfg.diffusion.schedule_type,
        cfg.diffusion.beta_start,
        cfg.diffusion.beta_end,
        cfg.diffusion.time_num
    )
    model = Model(
        cfg,
        betas,
        cfg.diffusion.loss_type,
        cfg.diffusion.model_mean_type,
        cfg.diffusion.model_var_type,
        cfg.model.width_mult,
        cfg.model.vox_res_mult
    )

    checkpoint_path = cfg.inference.model_ckpt
    assert os.path.isfile(checkpoint_path), f"Checkpoint not found: {checkpoint_path}"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(remove_module_prefix(ckpt['model_state']))
    model.eval()

    if cfg.inference.mesh_recon.proceed_mesh_recon:

        from mesh_recon.src.model import Encode2Points
        from mesh_recon.src.dpsr import DPSR
        from mesh_recon.src.utils import (load_model_manual, load_config, export_generated_mesh_crowns)


        sap_cfg = load_config(cfg.inference.mesh_recon.sap_tooth_cfg, cfg.inference.mesh_recon.sap_default_cfg)
        sap_model = Encode2Points(sap_cfg).to(device)
        load_model_manual(torch.load(cfg.inference.mesh_recon.sap_model_ckpt)['state_dict'], sap_model)
        dpsr = DPSR(res=(sap_cfg['model']['grid_res'], 
                            sap_cfg['model']['grid_res'], 
                            sap_cfg['model']['grid_res']), 
                        sig=sap_cfg['model']['psr_sigma']).to(device)
        sap_model.eval()



    patient_id_list = test_dataset.get_patient_ids()


    # patient_id_list: for test dataset 
    # {
    #     "TADP0046":[[11,12],[45,46,47]], # list of missing teeth FDIs to generate for patient TADP0046
    #     "TADP0047":[[33],[15,13,11,45,26,27]]
    # }

    for patient_id in patient_id_list:


        patient_save_dir = os.path.join(cfg.inference.save_dir, patient_id)
        os.makedirs(patient_save_dir, exist_ok=True)
        
        sample = test_dataset.sample_patient_by_patient_id(patient_id)
        pt_id = sample['patient_id']
        dentition_points = sample['dentition_points'].unsqueeze(0).to(device)  # (1, 28, 3, npoints)
        bound = sample['bounds_cyl'].unsqueeze(0).to(device)
        shift = sample['shift'].unsqueeze(0).numpy()
        scale = sample['scale'].unsqueeze(0).numpy()

        dentition_points_np = dentition_points.detach().cpu().numpy()
        missing_fdis_to_generate = patient_id_list[patient_id]
        
        if missing_fdis_to_generate != None:
            for missing_fdis in missing_fdis_to_generate:

                print(f"Generating for patient {patient_id} with missing FDIs: {missing_fdis}")

                scenario_save_dir = os.path.join(patient_save_dir, f'scenario_{missing_fdis}')
                os.makedirs(scenario_save_dir, exist_ok=True)
                
                latent_mask = torch.zeros_like(dentition_points[:, :, :1, :1]).to(device)
                missing_indices = [FDIS_ORDER.index(fdi) for fdi in missing_fdis]
                latent_mask[0, missing_indices, 0, 0] = 1

                obs_mask = torch.ones_like(latent_mask) - latent_mask

                data_dict = {
                    'x0': dentition_points,
                    'l_mask': latent_mask,
                    'o_mask': obs_mask,
                    'bound': bound
                }

                with torch.no_grad():
                    sample_out, _ = model.gen_samples(
                        model_kwargs=data_dict,
                        return_attn_weights=False
                    )
                    sample_out = sample_out.detach().cpu()
                    obs_mask = obs_mask.detach().cpu()
                    latent_mask = latent_mask.detach().cpu()

                    condition_points = dentition_points_np[0][obs_mask.view(-1).bool()]
                    generated_points = sample_out[0][latent_mask.view(-1).bool()].numpy()
                    ground_truth_points = dentition_points_np[0][latent_mask.view(-1).bool()]

                    export_dentition(
                        'condition',
                        scenario_save_dir,
                        condition_points,
                        shift=shift,
                        scale=scale,
                        combine_teeth=True
                    )
                    export_dentition(
                        'gen',
                        scenario_save_dir,
                        generated_points,
                        shift=shift,
                        scale=scale,
                        combine_teeth=True
                    )
                    export_dentition(
                        'gt',
                        scenario_save_dir,
                        ground_truth_points,
                        shift=shift,
                        scale=scale,
                        combine_teeth=True
                    )

                    if cfg.inference.mesh_recon.proceed_mesh_recon:
                        for i in range(generated_points.shape[0]):
                            fdi = FDIS_ORDER[missing_indices[i]]
                            generated_tooth_point = generated_points[i]
                            export_generated_mesh_crowns(sap_model, dpsr, sap_cfg, device,scenario_save_dir, shift.squeeze(0), scale.squeeze(0), fdi, generated_tooth_point)
                        