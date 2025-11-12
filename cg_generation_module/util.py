import os
import sys
import datetime
import random
from pathlib import Path
import numpy as np
from shutil import copyfile
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('agg')
import re
from collections import OrderedDict
import torch
import logging
import trimesh

logger = logging.getLogger()

def copy_source(file, output_dir):
    copyfile(file, os.path.join(output_dir, os.path.basename(file)))


def setup_logging(output_dir):
    log_format = logging.Formatter("%(asctime)s : %(message)s")
    logger = logging.getLogger()
    logger.handlers = []
    output_file = os.path.join(output_dir, 'output.log')
    file_handler = logging.FileHandler(output_file)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    err_handler = logging.StreamHandler(sys.stderr)
    err_handler.setFormatter(log_format)
    logger.addHandler(err_handler)
    logger.setLevel(logging.INFO)

    return logger


def get_output_dir(prefix, exp_id):
    t = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    output_dir = os.path.join(prefix, 'output/' + exp_id, t)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir


def set_seed(seed):
    if seed is None:
        seed = random.randint(1, 10000)
    print("Random Seed: ", seed)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def setup_output_subdirs(output_dir, *subfolders):
    output_subdirs = output_dir
    try:
        os.makedirs(output_subdirs)
    except OSError:
        pass

    subfolder_list = []
    for sf in subfolders:
        curr_subf = os.path.join(output_subdirs, sf)
        try:
            os.makedirs(curr_subf)
        except OSError:
            pass
        subfolder_list.append(curr_subf)

    return subfolder_list



def export_dentition(id, dir, points, shift, scale, combine_teeth=False):

    # points: (nT, 3, 1024)
    
    os.makedirs(dir, exist_ok=True)
    points = points.transpose(0,2,1)
    points = points * scale + shift  

    if combine_teeth or points.shape[0] == 1:
        points = points.reshape(-1,3)
        save_path = os.path.join(dir, '{}.npy'.format(id))
        np.save(save_path, points)
        trimesh.PointCloud(points).export(save_path.replace('.npy', '.ply'))
    else:
        for i in range(points.shape[0]):
            points_tooth = points[i].reshape(-1,3)
            save_path = os.path.join(dir, '{}_{}.npy'.format(id, i))
            np.save(save_path, points_tooth)
            trimesh.PointCloud(points_tooth).export(save_path.replace('.npy', '.ply'))


def export_to_pc_batch(keys, dir, pcs, colors=None):
    Path(dir).mkdir(parents=True, exist_ok=True)
    for i, xyz in enumerate(pcs):
        if colors is None:
            color = None
        else:
            color = colors[i]

        visualize_pointcloud(points = xyz, out_file=os.path.join(dir, 'sample_{}_{}.png'.format(keys[i],str(i))))
        np.save(os.path.join(dir, 'sample_{}_{}.npy'.format(keys[i],str(i))), xyz)
    

def visualize_pointcloud(points, normals=None, out_file=None, show=False, elev=30, azim=225):
    r''' Visualizes point cloud data.
    Args:
        points (tensor): point data
        normals (tensor): normal data (if existing)
        out_file (string): output file
        show (bool): whether the plot should be shown
    '''
    # Create plot
    fig = plt.figure(figsize=(40,40))
    # ax = fig.gca(projection=Axes3D.name)
    ax =  fig.add_subplot(projection = '3d')
    ax.scatter(points[:, 2], points[:, 0], points[:, 1])
    if normals is not None:
        ax.quiver(
            points[:, 2], points[:, 0], points[:, 1],
            normals[:, 2], normals[:, 0], normals[:, 1],
            length=0.1, color='k'
        )
    ax.set_xlabel('Z')
    ax.set_ylabel('X')
    ax.set_zlabel('Y')
    # ax.set_xlim(-0.5, 0.5)
    # ax.set_ylim(-0.5, 0.5)
    # ax.set_zlim(-0.5, 0.5)
    ax.view_init(elev=elev, azim=azim)
    if out_file is not None:
        plt.savefig(out_file)
    if show:
        plt.show()
    plt.close(fig)



def remove_module_prefix(state_dict):

    new_state_dict = OrderedDict()
    pattern = re.compile('module.')

    for k,v in state_dict.items():
        if re.search("module", k):
            new_state_dict[re.sub(pattern, '', k)] = v    
        else:
            new_state_dict[k] = v

    return new_state_dict