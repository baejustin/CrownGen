import os
import numpy as np
import torch as th
from glob import glob
import json
from torchvision import transforms
import pytorch_lightning as pl
from dataset.dentition_aug import *

FDIS_ORDER=[17,47,16,46,15,45,14,44,13,43,12,42,11,41,21,31,22,32,23,33,24,34,25,35,26,36,27,37]


class DentitionDataset(th.utils.data.Dataset):
    def __init__(self, 
                 path, 
                 mode = 'train', # train, val, or test
                 aug_transforms = None,
                 norm_mode = 'fixed',
                 tooth_npoints = 512):
        super().__init__()
        self.mode = mode
        self.aug_transforms = aug_transforms
        self.tooth_npoints = tooth_npoints
        self.norm_mode = norm_mode

        self.data_path = os.path.join(path, 'dentition')
        self.splits_path = os.path.join(path, 'splits_boundpred.json')

        self.preloaded_dentition = {}

        self.pid_to_idx_hash = {}

        self.patient_id_list = json.load(open(self.splits_path, 'r'))[self.mode]
        print('Preloading dentition...')

        for idx, patient_id in enumerate(self.patient_id_list): 
            self.pid_to_idx_hash[patient_id] = idx
            self.preloaded_dentition[idx] = {'patient_id':patient_id, 'data':{}, 'bounds':{}, 'missing_fdis':[]}

            for vert_path in glob(os.path.join(self.data_path, patient_id, 'verts', '*')):
                teeth_name = os.path.basename(vert_path)
                fdi = int(teeth_name.split('_')[-1].replace('.npy','').replace('FDI',''))
                if fdi not in FDIS_ORDER:
                    continue

                self.preloaded_dentition[idx]['data'][fdi] = np.load(vert_path)

                bound_path = vert_path.replace('verts','boundary').replace('.npy','_boundary.json')
                bound_json = json.load(open(bound_path,'r'))['cylinder']
                self.preloaded_dentition[idx]['bounds'][fdi] = {
                    'center':np.array([bound_json['cx'],bound_json['cy'],bound_json['cz']]),
                    'size':np.array([bound_json['h'],bound_json['r']]),
                }
                
    def __len__(self):
        return len(self.preloaded_dentition)
    
    def get_patient_ids(self):
        return self.patient_id_list

    def normalize_dentition(self, dentition_points, bound_cylinder_center, bound_cyliner_size):

        if self.norm_mode == 'fixed':
            shift = np.array([[0.0,20.0,0.0]])
            scale = np.array([[10.5]])
        elif self.norm_mode == 'dynamic':
            pc_max = dentition_points.max(axis=(0,1))
            pc_min = dentition_points.min(axis=(0,1))
            shift = ((pc_min + pc_max) / 2).reshape(1, 3)
            scale = ((pc_max - pc_min).max().reshape(1, 1) / 2 ) /3

        dentition_points_norm = (dentition_points - shift) / scale
        bound_cylinder_center_norm = (bound_cylinder_center - shift) / scale
        bound_cyliner_size_norm = (bound_cyliner_size) / scale


        return dentition_points_norm, bound_cylinder_center_norm, bound_cyliner_size_norm, shift, scale


    def sample_patient_by_patient_id(self, patient_id: list):
        return self.__getitem__(self.pid_to_idx_hash[patient_id])

    def __getitem__(self, idx):
        
        dentition_data_dict = self.preloaded_dentition[idx]

        dentition_points = []
        bound_cylinder_center = []
        bound_cyliner_size = []

        for fdi in FDIS_ORDER:
            vert = dentition_data_dict['data'][fdi]
            random_index = np.random.randint(0, vert.shape[0], self.tooth_npoints)
            dentition_points.append(vert[random_index])

            bound_cylinder_center.append(dentition_data_dict['bounds'][fdi]['center'])
            bound_cyliner_size.append(dentition_data_dict['bounds'][fdi]['size'])


        dentition_points = np.array(dentition_points).reshape(len(FDIS_ORDER), self.tooth_npoints, 3)
        bound_cylinder_center = np.array(bound_cylinder_center)
        bound_cyliner_size = np.array(bound_cyliner_size)

        dentition_points_norm, bound_cylinder_center_norm, bound_cyliner_size_norm, shift, scale = self.normalize_dentition(dentition_points,
                                                                                                    bound_cylinder_center,
                                                                                                    bound_cyliner_size)
                                                                                                            
        if self.aug_transforms and self.mode == 'train':

            aug_out = self.aug_transforms({'d':dentition_points_norm, 'bcc':bound_cylinder_center_norm, 'bcs':bound_cyliner_size_norm})
            
            dentition_points_norm = aug_out['d']
            bound_cylinder_center_norm = aug_out['bcc']
            bound_cyliner_size_norm = aug_out['bcs']
 

        out = {
            'patient_id': dentition_data_dict['patient_id'], 
            'dentition_points': th.from_numpy(dentition_points_norm).transpose(1,2).float(), 
            'bounds_cyl':th.from_numpy(np.concatenate([bound_cylinder_center_norm, bound_cyliner_size_norm], 1)).float(),
            'shift':th.from_numpy(shift).float(),
            'scale':th.from_numpy(scale).float()
        }

        return out


class DentitionDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.bs = cfg.data.batch_size
        self.datapath = cfg.data.path
        self.num_workers = cfg.data.num_workers
        self.tooth_npoints = cfg.data.tooth_npoints
        self.max_missing_teeth = cfg.data.max_missing_teeth

        self.transforms = transforms.Compose(
                                    [   
                                        RandomMirror(p=0.5),
                                        RandomScale(scale=[0.95, 1.05], p=0.75),
                                        ShufflePoint(p=0.9)
                                    ] )

    def setup(self, stage=None):

        self.train_dataset = DentitionDataset(path=self.datapath, mode='train', tooth_npoints=self.tooth_npoints, aug_transforms=self.transforms)
        self.val_dataset = DentitionDataset(path=self.datapath, mode='val', tooth_npoints=self.tooth_npoints)


    def train_dataloader(self):
        return th.utils.data.DataLoader(self.train_dataset, batch_size=self.bs,shuffle=True,num_workers=int(self.num_workers), persistent_workers=True, drop_last=False)
    def val_dataloader(self):
        return th.utils.data.DataLoader(self.val_dataset, batch_size=self.bs,shuffle=False,num_workers=int(self.num_workers), persistent_workers=True)
    

