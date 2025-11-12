

import os
import numpy as np
import torch as th
import json
from glob import glob


FDIS_ORDER=[17,47,16,46,15,45,14,44,13,43,12,42,11,41,21,31,22,32,23,33,24,34,25,35,26,36,27,37]


def subsample_occlusal(pcd, num_points, prefer_high_z=True):

    z_vals = pcd[:, 2]
    epsilon = 1e-8
    
    if prefer_high_z:
        min_z = np.min(z_vals)
        weight = ((z_vals - min_z)**6 + epsilon) 
    else:
        max_z = np.max(z_vals)
        weight = ((max_z - z_vals)**6 + epsilon) 
    
    p = weight / np.sum(weight)
    random_sample_index = np.random.choice(len(pcd), size=num_points, replace=False, p=p)
    return random_sample_index


class DentitionDataset(th.utils.data.Dataset):
    def __init__(self, 
                 path, 
                 mode = 'train',
                 aug_transforms = None,
                 norm_mode = 'fixed',
                 occlusal_subsampling = True,
                 boundary_size_scale = 1.0,
                 tooth_npoints = 1024):
        super().__init__()

        self.mode = mode
        self.aug_transforms = aug_transforms
        self.tooth_npoints = tooth_npoints
        self.occlusal_subsampling = occlusal_subsampling
        self.norm_mode = norm_mode

        self.data_path = os.path.join(path, 'dentition')
        self.splits_path = os.path.join(path, 'splits_gen.json')

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


    def sample_patient(self, sample_batch_size: int):
        indexes = np.random.randint(0, len(self), sample_batch_size)

        dentition_list = []
        for idx in indexes:
            dentition_list.append(self.__getitem__(idx))
        return dentition_list

    def sample_patient_by_patient_id(self, patient_id: list):
        return self.__getitem__(self.pid_to_idx_hash[patient_id])

    def __getitem__(self, idx):
        
        dentition_data_dict = self.preloaded_dentition[idx]
  

        dentition_points = []
        bound_cylinder_center = []
        bound_cyliner_size = []

        for fdi in FDIS_ORDER:
            vert = dentition_data_dict['data'][fdi]

            if self.occlusal_subsampling:

                if fdi < 30: #upper
                    random_index = subsample_occlusal(vert, self.tooth_npoints, prefer_high_z=False)
                else: # lower
                    random_index = subsample_occlusal(vert, self.tooth_npoints, prefer_high_z=True)

            else:
                random_index = np.random.randint(0, vert.shape[0], self.tooth_npoints)


            dentition_points.append(vert[random_index])

            bound_cylinder_center.append(dentition_data_dict['bounds'][fdi]['center'])
            bound_cyliner_size.append(dentition_data_dict['bounds'][fdi]['size'])



        dentition_points = np.array(dentition_points).reshape(len(FDIS_ORDER), self.tooth_npoints, 3)
        bound_cylinder_center = np.array(bound_cylinder_center)
        bound_cyliner_size = np.array(bound_cyliner_size)

        assert bound_cylinder_center.shape == (len(FDIS_ORDER), 3)
        assert bound_cyliner_size.shape == (len(FDIS_ORDER), 2)

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

