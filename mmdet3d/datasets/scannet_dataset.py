import numpy as np
import tempfile
import warnings
import os
import pickle as pkl
from os import path as osp
from mmdet3d.core import show_result, show_seg_result
from mmdet3d.core.bbox import DepthInstance3DBoxes
from mmdet3d.ops import points_in_boxes_cpu
from mmdet.datasets import DATASETS
from mmseg.datasets import DATASETS as SEG_DATASETS
from .custom_3d import Custom3DDataset
from .custom_3d_seg import Custom3DSegDataset
from .pipelines import Compose
import torch


@DATASETS.register_module()
class ScanNetDataset(Custom3DDataset):
    r"""ScanNet Dataset for Detection Task.

    This class serves as the API for experiments on the ScanNet Dataset.

    Please refer to the `github repo <https://github.com/ScanNet/ScanNet>`_
    for data downloading.

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        box_type_3d (str, optional): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'Depth' in this dataset. Available options includes

            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        filter_empty_gt (bool, optional): Whether to filter empty GT.
            Defaults to True.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
    """
    CLASSES = ('cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
               'bookshelf', 'picture', 'counter', 'desk', 'curtain',
               'refrigerator', 'showercurtrain', 'toilet', 'sink', 'bathtub',
               'garbagebin')

    def __init__(self,
                 data_root,
                 ann_file,
                 pipeline=None,
                 classes=None,
                 modality=None,
                 box_type_3d='Depth',
                 filter_empty_gt=True,
                 test_mode=False):
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode)

        gt_base_filename = os.path.join(self.data_root, 'scannet_gt_base_no_instance_segmentation.pkl')
        if os.path.exists(gt_base_filename):
            self.gt_base = pkl.load(open(gt_base_filename, 'rb'))
        else:
            self.gt_base = self.create_base()
            with open(gt_base_filename, 'wb') as gt_f:
                pkl.dump(self.gt_base, gt_f)

    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`DepthInstance3DBoxes`): \
                    3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - pts_instance_mask_path (str): Path of instance masks.
                - pts_semantic_mask_path (str): Path of semantic masks.
                - axis_align_matrix (np.ndarray): Transformation matrix for \
                    global scene alignment.
        """
        # Use index to get the annos, thus the evalhook could also use this api
        info = self.data_infos[index]
        if info['annos']['gt_num'] != 0:
            gt_bboxes_3d = info['annos']['gt_boxes_upright_depth'].astype(
                np.float32)  # k, 6
            gt_labels_3d = info['annos']['class'].astype(np.long)
        else:
            gt_bboxes_3d = np.zeros((0, 6), dtype=np.float32)
            gt_labels_3d = np.zeros((0, ), dtype=np.long)

        # to target box structure
        gt_bboxes_3d = DepthInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            with_yaw=False,
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)

        pts_instance_mask_path = osp.join(self.data_root,
                                          info['pts_instance_mask_path'])
        pts_semantic_mask_path = osp.join(self.data_root,
                                          info['pts_semantic_mask_path'])

        axis_align_matrix = self._get_axis_align_matrix(info)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            pts_instance_mask_path=pts_instance_mask_path,
            pts_semantic_mask_path=pts_semantic_mask_path,
            axis_align_matrix=axis_align_matrix)
        return anns_results

    def prepare_test_data(self, index):
        """Prepare data for testing.

        We should take axis_align_matrix from self.data_infos since we need \
            to align point clouds.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Testing data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        # take the axis_align_matrix from data_infos
        input_dict['ann_info'] = dict(
            axis_align_matrix=self._get_axis_align_matrix(
                self.data_infos[index]))
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        return example

    @staticmethod
    def _get_axis_align_matrix(info):
        """Get axis_align_matrix from info. If not exist, return identity mat.

        Args:
            info (dict): one data info term.

        Returns:
            np.ndarray: 4x4 transformation matrix.
        """
        if 'axis_align_matrix' in info['annos'].keys():
            return info['annos']['axis_align_matrix'].astype(np.float32)
        else:
            warnings.warn(
                'axis_align_matrix is not found in ScanNet data info, please '
                'use new pre-process scripts to re-generate ScanNet data')
            return np.eye(4).astype(np.float32)

    def _build_default_pipeline(self):
        """Build the default pipeline for this dataset."""
        pipeline = [
            dict(
                type='LoadPointsFromFile',
                coord_type='DEPTH',
                shift_height=False,
                load_dim=6,
                use_dim=[0, 1, 2]),
            dict(type='GlobalAlignment', rotation_axis=2),
            dict(
                type='DefaultFormatBundle3D',
                class_names=self.CLASSES,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ]
        return Compose(pipeline)

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - file_name (str): Filename of point clouds.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        sample_idx = info['point_cloud']['lidar_idx']
        pts_filename = osp.join(self.data_root, info['pts_path'])

        input_dict = dict(
            pts_filename=pts_filename,
            sample_idx=sample_idx,
            file_name=pts_filename)

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos
            if self.filter_empty_gt and ~(annos['gt_labels_3d'] != -1).any():
                return None
        return input_dict

    def prepare_train_data(self, index):
        """Training data preparation.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Training data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        if self.filter_empty_gt and \
                (example is None or
                    ~(example['gt_labels_3d']._data != -1).any()):
            return None
        return example

    def show(self, results, out_dir, show=True, pipeline=None):
        """Results visualization.

        Args:
            results (list[dict]): List of bounding boxes results.
            out_dir (str): Output directory of visualization result.
            show (bool): Visualize the results online.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.
        """
        assert out_dir is not None, 'Expect out_dir, got none.'
        pipeline = self._get_pipeline(pipeline)
        for i, result in enumerate(results):
            data_info = self.data_infos[i]
            pts_path = data_info['pts_path']
            file_name = osp.split(pts_path)[-1].split('.')[0]
            points = self._extract_data(i, pipeline, 'points', load_annos=True).numpy()
            gt_bboxes = self.get_ann_info(i)['gt_bboxes_3d']
            gt_bboxes = gt_bboxes.corners.numpy() if len(gt_bboxes) else None
            gt_labels = self.get_ann_info(i)['gt_labels_3d']
            pred_bboxes = result['boxes_3d']
            pred_bboxes = pred_bboxes.corners.numpy() if len(pred_bboxes) else None
            pred_labels = result['labels_3d']
            show_result(points, gt_bboxes, gt_labels,
                        pred_bboxes, pred_labels, out_dir, file_name, False)

    def get_parameters(self, pcs, bboxes):
        '''
        get parameters noted as Equation 7 in Correlation Field for Boosting 3D Object Detection in Structured Scenes
        pc: a list, total length is N, each element is (#points, 3+color)
        bbox: [N, 7]
        this only works for single pc instance point clouds.
        '''
        param_s = bboxes[:, 3:6]
        param_cg = np.zeros((bboxes.shape[0], 3))

        pc_center = [np.mean(x[:, :3], axis=0) for x in pcs]
        pc_center = np.stack(pc_center, axis=0)
        param_cg = pc_center - bboxes[:, :3]
        return param_s, param_cg
    
    def _build_gtsampler_pipeline(self):
        """Build the gtsampler pipeline for this dataset."""
        pipeline = [
            dict(
                type='LoadPointsFromFile',
                coord_type='DEPTH',
                load_dim=6,
                use_dim=[0, 1, 2, 3, 4, 5]),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True,
                with_mask_3d=True,
                with_seg_3d=True),
            dict(type='GlobalAlignment', rotation_axis=2),
        ]
        return Compose(pipeline)

    def create_base(self):
        '''
        @scan_names: names for all the scans needed to build the data base 
        @res_base (output): the data base for all the proposals
            must contain:
            raw_points
            instance_bboxes
        '''
        res_base = {}
        mask_per_ins_list = []
        instance_bboxes_all = []
        instance_labels_all = []
        instance_semantics_all = []
        raw_points = []
        pipeline = self._build_gtsampler_pipeline()
        for idx, data_info in enumerate(self.data_infos):
            scan_name = data_info['point_cloud']['lidar_idx']
            instance_bboxes = np.load(os.path.join(self.data_root, 'scannet_instance_data', scan_name) + '_aligned_bbox.npy')
            
            keys = ['points', 'gt_bboxes_3d', 'gt_labels_3d', 'pts_instance_mask', 'pts_semantic_mask']
            data_list = self._extract_data(idx, pipeline, keys, load_annos=True)
            points_depth, gt_bboxes_3d_depth, gt_labels_3d, pts_instance_mask, pts_semantic_mask = data_list
            points = points_depth.tensor.numpy()
            gt_bboxes_3d = gt_bboxes_3d_depth.tensor.numpy()

            for i_ins, each_bbox in enumerate(instance_bboxes):
                # 将深度坐标系下的点云转换为LiDAR坐标系
                points_lidar = points.copy()
                points_lidar[:, [0, 1, 2]] = points[:, [2, 0, 1]]  # depth -> lidar: (x,y,z) -> (z,x,y)
                points_lidar[:, 1] = -points_lidar[:, 1]  # 翻转y轴方向
                
                # 将深度坐标系下的bbox转换为LiDAR坐标系
                bbox_lidar = each_bbox.copy()
                bbox_lidar[[0, 1, 2]] = each_bbox[[2, 0, 1]]  # 坐标中心点转换
                bbox_lidar[[3, 4, 5]] = each_bbox[[5, 3, 4]]  # bbox尺寸转换
                bbox_lidar[1] = -bbox_lidar[1]  # 翻转y轴方向
                bbox_for_check = bbox_lidar[np.newaxis, ...]  # [1, 7]
                point_mask = points_in_boxes_cpu(points_lidar[:, :3], bbox_for_check)
                point_mask = point_mask.squeeze()  # [N]
                
                semantic_mask = (pts_semantic_mask == each_bbox[-1])
                first_mask = np.logical_and(point_mask, semantic_mask)
                if np.sum(first_mask) <= 50:
                    continue
                
                mask_per_ins_list.append(point_mask)
                raw_points.append(points[np.where(point_mask)[0]]) # depth坐标系下的点云
                instance_bboxes_all.append(gt_bboxes_3d[i_ins]) # depth坐标系下的bbox
                instance_labels_all.append(gt_labels_3d[i_ins])
                instance_semantics_all.append(each_bbox[-1])
        
        res_base['raw_points_in_boxes'] = raw_points
        res_base['instance_bboxes'] = np.stack(instance_bboxes_all, axis=0)
        res_base['instance_labels'] = np.stack(instance_labels_all, axis=0)
        res_base['instance_semantics'] = np.stack(instance_semantics_all, axis=0)
        res_base['mask_per_instance'] = mask_per_ins_list

        # add parameters
        param_s, param_cg = self.get_parameters(raw_points, res_base['instance_bboxes'])
        res_base['param_s'] = param_s
        res_base['param_cg'] = param_cg
        return res_base

    def save_points_to_obj(self, points, save_path):
        """
        将点云数据保存为 OBJ 文件
        Args:
            points: numpy array 或 torch.Tensor，形状为 (N, 3)，包含点云的 x,y,z 坐标
            save_path: 字符串，保存文件的路径，以 .obj 结尾
        """
        # 如果输入是 torch.Tensor，转换为 numpy array
        if isinstance(points, torch.Tensor):
            points = points.cpu().numpy()
        
        with open(save_path, 'w') as f:
            # 写入顶点信息
            for point in points:
                f.write(f'v {point[0]} {point[1]} {point[2]}\n')

@DATASETS.register_module()
@SEG_DATASETS.register_module()
class ScanNetSegDataset(Custom3DSegDataset):
    r"""ScanNet Dataset for Semantic Segmentation Task.

    This class serves as the API for experiments on the ScanNet Dataset.

    Please refer to the `github repo <https://github.com/ScanNet/ScanNet>`_
    for data downloading.

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        palette (list[list[int]], optional): The palette of segmentation map.
            Defaults to None.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
        ignore_index (int, optional): The label index to be ignored, e.g. \
            unannotated points. If None is given, set to len(self.CLASSES).
            Defaults to None.
        scene_idxs (np.ndarray | str, optional): Precomputed index to load
            data. For scenes with many points, we may sample it several times.
            Defaults to None.
    """
    CLASSES = ('wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table',
               'door', 'window', 'bookshelf', 'picture', 'counter', 'desk',
               'curtain', 'refrigerator', 'showercurtrain', 'toilet', 'sink',
               'bathtub', 'otherfurniture')

    VALID_CLASS_IDS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28,
                       33, 34, 36, 39)

    ALL_CLASS_IDS = tuple(range(41))

    PALETTE = [
        [174, 199, 232],
        [152, 223, 138],
        [31, 119, 180],
        [255, 187, 120],
        [188, 189, 34],
        [140, 86, 75],
        [255, 152, 150],
        [214, 39, 40],
        [197, 176, 213],
        [148, 103, 189],
        [196, 156, 148],
        [23, 190, 207],
        [247, 182, 210],
        [219, 219, 141],
        [255, 127, 14],
        [158, 218, 229],
        [44, 160, 44],
        [112, 128, 144],
        [227, 119, 194],
        [82, 84, 163],
    ]

    def __init__(self,
                 data_root,
                 ann_file,
                 pipeline=None,
                 classes=None,
                 palette=None,
                 modality=None,
                 test_mode=False,
                 ignore_index=None,
                 scene_idxs=None):

        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=classes,
            palette=palette,
            modality=modality,
            test_mode=test_mode,
            ignore_index=ignore_index,
            scene_idxs=scene_idxs)

    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: annotation information consists of the following keys:

                - pts_semantic_mask_path (str): Path of semantic masks.
        """
        # Use index to get the annos, thus the evalhook could also use this api
        info = self.data_infos[index]

        pts_semantic_mask_path = osp.join(self.data_root,
                                          info['pts_semantic_mask_path'])

        anns_results = dict(pts_semantic_mask_path=pts_semantic_mask_path)
        return anns_results

    def _build_default_pipeline(self):
        """Build the default pipeline for this dataset."""
        pipeline = [
            dict(
                type='LoadPointsFromFile',
                coord_type='DEPTH',
                shift_height=False,
                use_color=True,
                load_dim=6,
                use_dim=[0, 1, 2, 3, 4, 5]),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=False,
                with_label_3d=False,
                with_mask_3d=False,
                with_seg_3d=True),
            dict(
                type='PointSegClassMapping',
                valid_cat_ids=self.VALID_CLASS_IDS,
                max_cat_id=np.max(self.ALL_CLASS_IDS)),
            dict(
                type='DefaultFormatBundle3D',
                with_label=False,
                class_names=self.CLASSES),
            dict(type='Collect3D', keys=['points', 'pts_semantic_mask'])
        ]
        return Compose(pipeline)

    def show(self, results, out_dir, show=True, pipeline=None):
        """Results visualization.

        Args:
            results (list[dict]): List of bounding boxes results.
            out_dir (str): Output directory of visualization result.
            show (bool): Visualize the results online.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.
        """
        assert out_dir is not None, 'Expect out_dir, got none.'
        pipeline = self._get_pipeline(pipeline)
        for i, result in enumerate(results):
            data_info = self.data_infos[i]
            pts_path = data_info['pts_path']
            file_name = osp.split(pts_path)[-1].split('.')[0]
            points, gt_sem_mask = self._extract_data(
                i, pipeline, ['points', 'pts_semantic_mask'], load_annos=True)
            points = points.numpy()
            pred_sem_mask = result['semantic_mask'].numpy()
            show_seg_result(points, gt_sem_mask,
                            pred_sem_mask, out_dir, file_name,
                            np.array(self.PALETTE), self.ignore_index, show)

    def get_scene_idxs(self, scene_idxs):
        """Compute scene_idxs for data sampling.

        We sample more times for scenes with more points.
        """
        # when testing, we load one whole scene every time
        if not self.test_mode and scene_idxs is None:
            raise NotImplementedError(
                'please provide re-sampled scene indexes for training')

        return super().get_scene_idxs(scene_idxs)

    def format_results(self, results, txtfile_prefix=None):
        r"""Format the results to txt file. Refer to `ScanNet documentation
        <http://kaldir.vc.in.tum.de/scannet_benchmark/documentation>`_.

        Args:
            outputs (list[dict]): Testing results of the dataset.
            txtfile_prefix (str | None): The prefix of saved files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: (outputs, tmp_dir), outputs is the detection results,
                tmp_dir is the temporal directory created for saving submission
                files when ``submission_prefix`` is not specified.
        """
        import mmcv

        if txtfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            txtfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None
        mmcv.mkdir_or_exist(txtfile_prefix)

        # need to map network output to original label idx
        pred2label = np.zeros(len(self.VALID_CLASS_IDS)).astype(np.int)
        for original_label, output_idx in self.label_map.items():
            if output_idx != self.ignore_index:
                pred2label[output_idx] = original_label

        outputs = []
        for i, result in enumerate(results):
            info = self.data_infos[i]
            sample_idx = info['point_cloud']['lidar_idx']
            pred_sem_mask = result['semantic_mask'].numpy().astype(np.int)
            pred_label = pred2label[pred_sem_mask]
            curr_file = f'{txtfile_prefix}/{sample_idx}.txt'
            np.savetxt(curr_file, pred_label, fmt='%d')
            outputs.append(dict(seg_mask=pred_label))

        return outputs, tmp_dir
