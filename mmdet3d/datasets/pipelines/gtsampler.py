import os
import random
import torch
import numpy as np
import pickle as pkl
import time

from mmdet.datasets.builder import PIPELINES
from mmdet3d.ops.pcdet_nms import pcdet_nms_utils

@PIPELINES.register_module()
class GTSampler(object):

    def __init__(self, using_instance_segmentation=False, data_root=None):
        self.sample_num_max = 15
        self.sample_iou_threshold = 0.2
        self.using_instance_segmentation = using_instance_segmentation
        self.data_root = data_root
        if using_instance_segmentation:
            gt_base_filename = os.path.join(self.data_root, 'scannet_gt_base.pkl')
        else:
            gt_base_filename = os.path.join(self.data_root, 'scannet_gt_base_no_instance_segmentation.pkl')
        self.gt_base_filename = gt_base_filename
        if os.path.exists(self.gt_base_filename):
            self.gt_base = pkl.load(open(self.gt_base_filename, 'rb'))
        else:
            print(f'{self.gt_base_filename} does not exist')
            self.gt_base = None

        self.base_bboxes = torch.from_numpy(self.gt_base['instance_bboxes']).to(torch.float32)
        self.base_raw_pc = self.gt_base['raw_points_in_boxes']
        self.base_labels = self.gt_base['instance_labels']
        self.base_semantics = self.gt_base['instance_semantics']
        self.iou2 = pcdet_nms_utils.boxes_bev_iou_cpu(self.base_bboxes, self.base_bboxes).numpy()

    def __call__(self, results):
        if self.gt_base is None:
            self.gt_base = pkl.load(open(self.gt_base_filename, 'rb'))
        self.sample_from_base(results)
        return results
    
    def sample_from_base(self, results):
        '''
        '''
        input_pc = results['points']
        input_bbox = results['gt_bboxes_3d'].tensor
        input_labels = results['gt_labels_3d']
        input_instance_labels = results['pts_instance_mask']
        input_semantic_labels = results['pts_semantic_mask']

        iou1 = pcdet_nms_utils.boxes_bev_iou_cpu(self.base_bboxes, input_bbox).numpy()
        self.iou2[range(self.iou2.shape[0]), range(self.iou2.shape[0])] = 0
        if iou1.shape[1] > 0:
            iou1_mask = (np.max(iou1, axis=1) == 0)
            iou2_mask = (np.max(self.iou2, axis=1) < self.sample_iou_threshold)
            mask_final = np.logical_and(iou1_mask, iou2_mask)
        else:
            iou2_mask = (np.max(self.iou2, axis=1) < self.sample_iou_threshold)
            mask_final = iou2_mask

        indexes_final = np.where(mask_final)[0]
        sample_num = random.randint(0, self.sample_num_max)
        if len(indexes_final) == 0 or sample_num == 0:
            return results
        elif len(indexes_final) < sample_num:
            pass
        else:
            indexes_final = np.random.choice(indexes_final, (sample_num), replace=False)
        
        res_point_clouds = input_pc.tensor.numpy()
        res_bboxes = input_bbox.numpy()
        res_instance_labels = input_instance_labels
        res_semantic_labels = input_semantic_labels
        res_labels = input_labels
        for indx in indexes_final:
            res_point_clouds = np.concatenate([res_point_clouds, self.base_raw_pc[indx][:, :res_point_clouds.shape[1]]], axis=0)
            res_bboxes = np.concatenate([res_bboxes, self.base_bboxes[indx][np.newaxis, ...]], axis=0)
            tem_array = np.ones(self.base_raw_pc[indx].shape[0])
            res_instance_labels = np.concatenate([res_instance_labels, tem_array * np.max(res_instance_labels + 1)])
            res_semantic_labels = np.concatenate([res_semantic_labels, tem_array * self.base_semantics[indx]])
            res_labels = np.concatenate([res_labels, np.array([self.base_labels[indx]])], axis=0)
        
        results['points'].tensor = torch.from_numpy(res_point_clouds)
        results['gt_bboxes_3d'].tensor = torch.from_numpy(res_bboxes)
        results['gt_labels_3d'] = res_labels.astype(np.int64)
        results['pts_instance_mask'] = res_instance_labels.astype(np.int64)
        results['pts_semantic_mask'] = res_semantic_labels.astype(np.int64)
        return results