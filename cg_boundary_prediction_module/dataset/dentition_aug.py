import random
import numpy as np


class RandomScale(object): 
    def __init__(self, scale=None, anisotropic=False, p=0.5):
        self.scale = scale if scale is not None else [0.95, 1.05]
        self.anisotropic = anisotropic
        self.p = p

    def __call__(self, dentition_data):


        if random.random() < self.p:


            scale = np.random.uniform(
                self.scale[0], self.scale[1], 3 if self.anisotropic else 1
            )

            dentition_data['d'] = dentition_data['d'] * scale
            dentition_data['bcc'] = dentition_data['bcc'] * scale
            dentition_data['bcs'] = dentition_data['bcs'] * scale


        return dentition_data


class ShufflePoint(object): 
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, dentition_data):
        
        if random.random() < self.p:

            dentition_points = dentition_data['d']


            num_teeth, num_points, _ = dentition_points.shape

            for i in range(num_teeth):
                shuffle_index = np.arange(num_points)
                np.random.shuffle(shuffle_index)
                dentition_points[i] = dentition_points[i][shuffle_index]

            dentition_data['d'] = dentition_points

        return dentition_data



class RandomJitter(object): 
    def __init__(self, sigma=0.01, clip=0.05, p=0.5):
        assert clip > 0
        self.sigma = sigma
        self.clip = clip
        self.p = p

    def __call__(self, dentition_data):

        if random.random() < self.p:

            dentition_points = dentition_data['d']

            num_teeth, num_points, _ = dentition_points.shape

            for i in range(num_teeth):

                jitter = np.clip(
                    self.sigma * np.random.randn(num_points, 3),
                    -self.clip,
                    self.clip,
                )

                dentition_points[i] = dentition_points[i] + jitter

            dentition_data['d'] = dentition_points

        return dentition_data






class RandomMirror(object):  
    def __init__(self, p=0.5, center=None):
        self.center=center
        self.p = p 
        self.new_fdi_order = [26, 27, 24, 25, 22, 23, 20, 21, 18, 19, 16, 17, 14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1]


    def __call__(self, dentition_data):

        
        if random.random() < self.p:


            dentition_points = dentition_data['d']
            bound_cylinder_center = dentition_data['bcc']
            bound_cyliner_size = dentition_data['bcs']


            dentition_points_ = dentition_points.copy()
            bound_cylinder_center_ = bound_cylinder_center.copy()
            bound_cyliner_size_ = bound_cyliner_size.copy()

            if self.center is None:
                x_min, y_min, z_min = dentition_points_.min(axis=(0,1))
                x_max, y_max, z_max = dentition_points_.max(axis=(0,1))
                center = [(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2]
            else:
                center = self.center

            dentition_points_ = dentition_points_ - center
            dentition_points_[..., 0] = -dentition_points_[..., 0]
            dentition_points_ = dentition_points_ + center

            bound_cylinder_center_ = bound_cylinder_center_ - center
            bound_cylinder_center_[..., 0] = -bound_cylinder_center_[..., 0]
            bound_cylinder_center_ = bound_cylinder_center_ + center


            dentition_points_ = dentition_points_[self.new_fdi_order]
            bound_cylinder_center_ = bound_cylinder_center_[self.new_fdi_order]
            bound_cyliner_size_ = bound_cyliner_size_[self.new_fdi_order]

            dentition_data['d'] = dentition_points_
            dentition_data['bcc'] = bound_cylinder_center_
            dentition_data['bcs'] = bound_cyliner_size_

        return dentition_data