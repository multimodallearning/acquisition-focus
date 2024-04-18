import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime as dt


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def get_volume_views(volume):
    volume = volume.squeeze().__ge__(0.5)
    fig = plt.figure()
    ax = fig.gca(projection=Axes3D.name)
    ax.set_aspect('equal')
    ax.voxels(volume, edgecolor="k")

    fig.canvas.draw()
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return img


def save_test_volumes_as_np(cfg, volume, sample_id, epoch_num):
    img_dir = os.path.join(cfg.DIR.OUT_PATH, 'images')
    test_case_path = os.path.join(img_dir, 'test')
    save_path = os.path.join(test_case_path, str(sample_id) + os.sep)

    if not os.path.isdir(img_dir):
        os.mkdir(img_dir)
    if not os.path.isdir(test_case_path):
        os.mkdir(test_case_path)
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    np.save(save_path + 'epoch_' + str(epoch_num), volume.cpu().numpy())


def get_loss_function(loss_func):
    if loss_func.lower() == 'bceloss':
        loss_func = torch.nn.BCELoss()
    elif loss_func.lower() == 'iou':
        loss_func = net_loss.IoULoss()
    elif loss_func.lower() == 'focalloss':
        loss_func = net_loss.FocalLoss()
    elif loss_func.lower() == 'tverskyloss':
        loss_func = net_loss.TverskyLoss()
    elif loss_func.lower() == 'focaltverskyloss':
        loss_func = net_loss.FocalTverskyLoss()
    else:
        raise Exception('[FATAL] %s No matching loss function available for: %s. voxels' % (
            dt.now(), loss_func))
    return loss_func
