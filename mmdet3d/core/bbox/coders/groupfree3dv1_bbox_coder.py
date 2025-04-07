import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from mmdet.core.bbox.builder import BBOX_CODERS
from .partial_bin_based_bbox_coder import PartialBinBasedBBoxCoder

class Integral(nn.Module):
    """A fixed layer for calculating integral result from distribution.

    This layer calculates the target location by :math: `sum{P(y_i) * y_i}`,
    P(y_i) denotes the softmax vector that represents the discrete distribution
    y_i denotes the discrete set, usually {0, 1, 2, ..., reg_max}

    Args:
        reg_max (int): The maximal value of the discrete set. Default: 16. You
            may want to reset it according to your new dataset or related
            settings.
    """

    def __init__(self, reg_max=16):
        super(Integral, self).__init__()
        self.reg_max = reg_max
        self.register_buffer('project',
                             torch.linspace(0, self.reg_max, self.reg_max + 1)/self.reg_max)

    def forward(self, x):
        """Forward feature from the regression head to get integral result of
        bounding box location.

        Args:
            x (Tensor): Features of the regression head, shape (N, 6*(n+1)),
                n is self.reg_max.

        Returns:
            x (Tensor): Integral result of box locations, i.e., distance
                offsets from the box center in four directions, shape (N, 6).
        """
        x = F.softmax(x.reshape(-1, self.reg_max + 1), dim=1)
        x = F.linear(x, self.project.type_as(x)).reshape(-1, 6)
        return x

class AngleIntegral(nn.Module):
    """A fixed layer for calculating integral result from distribution.

    This layer calculates the target location by :math: `sum{P(y_i) * y_i}`,
    P(y_i) denotes the softmax vector that represents the discrete distribution
    y_i denotes the discrete set, usually {0, 1, 2, ..., reg_max}

    Args:
        reg_max (int): The maximal value of the discrete set. Default: 16. You
            may want to reset it according to your new dataset or related
            settings.
    """

    def __init__(self, reg_max=16):
        super(AngleIntegral, self).__init__()
        self.reg_max = reg_max
        self.register_buffer('project',
                             torch.linspace(0, self.reg_max, self.reg_max + 1)/self.reg_max)

    def forward(self, x):
        """Forward feature from the regression head to get integral result of
        bounding box location.

        Args:
            x (Tensor): Features of the regression head, shape (N, (n+1)),
                n is self.reg_max.

        Returns:
            x (Tensor): Integral result of box locations, i.e., distance
                offsets from the box center in four directions, shape (N, 1).
        """
        x = F.softmax(x.reshape(-1, self.reg_max + 1), dim=1)
        x = F.linear(x, self.project.type_as(x)).reshape(-1, 1)
        return x



@BBOX_CODERS.register_module()
class GroupFree3DBBoxCoderV1(PartialBinBasedBBoxCoder):
    """Modified partial bin based bbox coder for GroupFree3D.

    Args:
        num_dir_bins (int): Number of bins to encode direction angle.
        num_sizes (int): Number of size clusters.
        mean_sizes (list[list[int]]): Mean size of bboxes in each class.
        with_rot (bool): Whether the bbox is with rotation. Defaults to True.
        size_cls_agnostic (bool): Whether the predicted size is class-agnostic.
            Defaults to True.
    """

    def __init__(self,
                 num_dir_bins,
                 num_sizes,
                 mean_sizes,
                 reg_max,
                 reg_topk,
                 sizes=[3.0,3.0,2.5],
                 with_rot=True,
                 size_cls_agnostic=True):
        super(GroupFree3DBBoxCoderV1, self).__init__(
            num_dir_bins=num_dir_bins,
            num_sizes=num_sizes,
            mean_sizes=mean_sizes,
            with_rot=with_rot)
        self.reg_max = reg_max
        self.reg_topk = reg_topk
        self.sizes = sizes
        self.head_reg_outs = 12
        self.n_reg_outs = 6 * (self.reg_max + 1)
        self.size_cls_agnostic = size_cls_agnostic
        self.integral = Integral(self.reg_max)
        self.angle_integral = AngleIntegral(self.head_reg_outs - 1)

    def encode(self, gt_bboxes_3d, gt_labels_3d):
        """Encode ground truth to prediction targets.

        Args:
            gt_bboxes_3d (BaseInstance3DBoxes): Ground truth bboxes \
                with shape (n, 7).
            gt_labels_3d (torch.Tensor): Ground truth classes.

        Returns:
            tuple: Targets of center, size and direction.
        """
        # generate center target
        center_target = gt_bboxes_3d.gravity_center

        # generate bbox size target
        size_target = gt_bboxes_3d.dims
        size_class_target = gt_labels_3d.to(torch.int64)
        size_res_target = gt_bboxes_3d.dims - gt_bboxes_3d.tensor.new_tensor(
            self.mean_sizes)[size_class_target]

        # generate dir target
        box_num = gt_labels_3d.shape[0]
        if self.with_rot:
            (dir_class_target,
             dir_res_target) = self.angle2class(gt_bboxes_3d.yaw)
        else:
            dir_class_target = gt_labels_3d.new_zeros(box_num)
            dir_res_target = gt_bboxes_3d.tensor.new_zeros(box_num)

        return (center_target, size_target, size_class_target, size_res_target,
                dir_class_target, dir_res_target)

    def split_pred(self, cls_preds, reg_preds, base_xyz, prefix=''):
        """Split predicted features to specific parts.

        Args:
            cls_preds (torch.Tensor): Class predicted features to split.
            reg_preds (torch.Tensor): Regression predicted features to split.
            base_xyz (torch.Tensor): Coordinates of points.
            prefix (str): Decode predictions with specific prefix.
                Defaults to ''.

        Returns:
            dict[str, torch.Tensor]: Split results.
        """
        results = {}

        cls_preds_trans = cls_preds.transpose(2, 1)
        reg_preds_trans = reg_preds.transpose(2, 1)
        B, proposal_num = reg_preds_trans.shape[:2]

        surface_pred_res = self.integral(reg_preds_trans[..., :self.n_reg_outs]).reshape(B, proposal_num, -1)
        scale_x = torch.exp(reg_preds_trans[..., self.n_reg_outs + 0])
        scale_y = torch.exp(reg_preds_trans[..., self.n_reg_outs + 1])
        scale_z = torch.exp(reg_preds_trans[..., self.n_reg_outs + 2])
        # scale_x = torch.ones_like(reg_preds_trans[..., self.n_reg_outs + 0]) * torch.tensor(self.sizes[0])
        # scale_y = torch.ones_like(reg_preds_trans[..., self.n_reg_outs + 0]) * torch.tensor(self.sizes[1])
        # scale_z = torch.ones_like(reg_preds_trans[..., self.n_reg_outs + 0]) * torch.tensor(self.sizes[2])
        x1 = base_xyz[..., 0] - surface_pred_res[..., 0] * scale_x     
        y1 = base_xyz[..., 1] - surface_pred_res[..., 1] * scale_y
        z1 = base_xyz[..., 2] - surface_pred_res[..., 2] * scale_z
        x2 = base_xyz[..., 0] + surface_pred_res[..., 3] * scale_x
        y2 = base_xyz[..., 1] + surface_pred_res[..., 4] * scale_y
        z2 = base_xyz[..., 2] + surface_pred_res[..., 5] * scale_z
        results[f'{prefix}surface_pred'] = torch.stack((x1, y1, z1, x2, y2, z2), dim=-1).contiguous()
        results[f'{prefix}surface_scale'] = torch.stack((scale_x, scale_y, scale_z, scale_x, scale_y, scale_z), dim=-1).contiguous()
        
        # norm = torch.pow(torch.pow(reg_preds_trans[..., self.n_reg_outs + 0], 2) + torch.pow(reg_preds_trans[..., self.n_reg_outs + 1], 2), 0.5)
        # sin = reg_preds_trans[..., self.n_reg_outs + 0] / norm
        # cos = reg_preds_trans[..., self.n_reg_outs + 1] / norm
        angles = self.angle_integral(reg_preds_trans[..., self.n_reg_outs + 3:]).reshape(B, proposal_num) * 2 * torch.pi
        angles[angles > torch.pi] -= 2 * torch.pi

        results[f'{prefix}bbox_preds'] = torch.stack((
            (x1 + x2)/2.0,
            (y1 + y2)/2.0,
            (z1 + z2)/2.0,
            x2 - x1,
            y2 - y1,
            z2 - z1,
            angles
        ), dim=-1).contiguous()

        probs = reg_preds[:, :self.n_reg_outs, :]
        prob = F.softmax(probs.reshape(B, 6, self.reg_max+1, proposal_num), dim=2)
        results[f'{prefix}bbox_probs'] = prob

        results[f'{prefix}center'] = results[f'{prefix}bbox_preds'][..., :3].contiguous()
        results[f'{prefix}size'] = results[f'{prefix}bbox_preds'][..., 3:6].contiguous()
        results[f'{prefix}heading'] = results[f'{prefix}bbox_preds'][..., -1].contiguous()
        # decode objectness score
        # Group-Free-3D objectness output shape (batch, proposal, 1)
        results[f'{prefix}obj_scores'] = cls_preds_trans[..., :1].contiguous()
        # decode semantic score
        results[f'{prefix}sem_scores'] = cls_preds_trans[..., 1:].contiguous()

        return results
