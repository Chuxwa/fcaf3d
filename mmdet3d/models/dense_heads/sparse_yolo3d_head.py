import torch
from torch import nn
import MinkowskiEngine as ME
from mmdet.core import multi_apply, reduce_mean, build_assigner, BaseAssigner
from mmdet.core.bbox.builder import BBOX_ASSIGNERS
from mmdet.models.builder import HEADS, build_loss
from mmcv.cnn import bias_init_with_prob

from mmdet3d.core.bbox.structures import rotation_3d_in_axis
from mmdet3d.core.post_processing import aligned_3d_nms, box3d_multiclass_nms


class SparseYolo3DHead(nn.Module):
    def __init__(self,
                 n_classes,
                 n_channels,
                 n_convs,
                 n_reg_outs,
                 voxel_size,
                 assigner,
                 loss_bbox=dict(type='AxisAlignedIoULoss', loss_weight=1.0),
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 train_cfg=None,
                 test_cfg=None):
        super().__init__()
        self.n_classes = n_classes
        self.voxel_size = voxel_size
        self.assigner = build_assigner(assigner)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_cls = build_loss(loss_cls)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self._init_layers(n_channels, n_convs, n_reg_outs)

    @staticmethod
    def _make_block(in_channels, out_channels):
        return nn.Sequential(
            ME.MinkowskiConvolution(in_channels, out_channels, kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiELU()
        )

    def _init_layers(self, n_channels, n_convs, n_reg_outs):
        self.mlvl_reg_convs = nn.ModuleList([
            nn.Sequential(*[
                self._make_block(n_channels, n_channels)
                for _ in range(n_convs)
            ]) for _ in range(self.assigner.n_scales)
        ])
        self.mlvl_cls_convs = nn.ModuleList([
            nn.Sequential(*[
                self._make_block(n_channels, n_channels)
                for _ in range(n_convs)
            ]) for _ in range(self.assigner.n_scales)
        ])
        self.mlvl_reg_conv = nn.ModuleList([
            ME.MinkowskiConvolution(n_channels, n_reg_outs, kernel_size=1, dimension=3)
            for _ in range(self.assigner.n_scales)
        ])
        self.mlvl_cls_conv = nn.ModuleList([
            ME.MinkowskiConvolution(n_channels, self.n_classes, kernel_size=1, bias=True, dimension=3)
            for _ in range(self.assigner.n_scales)
        ])

    def init_weights(self):
        for module in list(self.mlvl_cls_convs.modules())[1:]:
            if type(module) == ME.MinkowskiConvolution:
                nn.init.normal_(module.kernel, std=.01)
        for module in list(self.mlvl_reg_convs.modules())[1:]:
            if type(module) == ME.MinkowskiConvolution:
                nn.init.normal_(module.kernel, std=.01)
        for module in list(self.mlvl_cls_conv.modules())[1:]:
            nn.init.normal_(module.kernel, std=.01)
            nn.init.constant_(module.bias, bias_init_with_prob(.01))
        for module in list(self.mlvl_reg_conv.modules())[1:]:
            nn.init.normal_(module.kernel, std=.01)

    def forward(self, x):
        return multi_apply(self.forward_single, x, self.mlvl_reg_convs, self.mlvl_cls_convs,
                           self.mlvl_reg_conv, self.mlvl_cls_conv)

    def loss(self,
             bbox_preds,
             cls_scores,
             points,
             gt_bboxes,
             gt_labels,
             img_metas):
        assert len(bbox_preds[0]) == len(cls_scores[0]) \
               == len(points[0]) == len(img_metas) == len(gt_bboxes) == len(gt_labels)

        loss_bbox, loss_cls = [], []
        for i in range(len(img_metas)):
            img_loss_bbox, img_loss_cls = self._loss_single(
                bbox_preds=[x[i] for x in bbox_preds],
                cls_scores=[x[i] for x in cls_scores],
                points=[x[i] for x in points],
                img_meta=img_metas[i],
                gt_bboxes=gt_bboxes[i],
                gt_labels=gt_labels[i]
            )
            loss_bbox.append(img_loss_bbox)
            loss_cls.append(img_loss_cls)
        return dict(
            loss_bbox=torch.mean(torch.stack(loss_bbox)),
            loss_cls=torch.mean(torch.stack(loss_cls))
        )

    # per image
    def _loss_single(self,
                     bbox_preds,
                     cls_scores,
                     points,
                     gt_bboxes,
                     gt_labels,
                     img_meta):
        bbox_targets, labels = self.assigner.assign(points, gt_bboxes, gt_labels)

        bbox_preds = torch.cat(bbox_preds)
        cls_scores = torch.cat(cls_scores)
        points = torch.cat(points)

        # skip background
        pos_inds = torch.nonzero(labels >= 0).squeeze(1)
        n_pos = torch.tensor(len(pos_inds), dtype=torch.float, device=bbox_preds.device)
        n_pos = max(reduce_mean(n_pos), 1.)
        loss_cls = self.loss_cls(cls_scores, labels, avg_factor=n_pos)
        pos_bbox_preds = bbox_preds[pos_inds]
        pos_bbox_targets = bbox_targets[pos_inds]

        if len(pos_inds) > 0:
            pos_points = points[pos_inds]
            loss_bbox = self.loss_bbox(
                self._bbox_pred_to_loss(pos_points, pos_bbox_preds),
                self._bbox_pred_to_loss(pos_points, pos_bbox_targets),
                avg_factor=n_pos
            )
        else:
            loss_bbox = pos_bbox_preds.sum()
        return loss_bbox, loss_cls

    def get_bboxes(self,
                   bbox_preds,
                   cls_scores,
                   points,
                   img_metas,
                   rescale=False):
        assert len(bbox_preds[0]) == len(cls_scores[0]) \
               == len(points[0]) == len(img_metas)
        results = []
        for i in range(len(img_metas)):
            result = self._get_bboxes_single(
                bbox_preds=[x[i] for x in bbox_preds],
                cls_scores=[x[i] for x in cls_scores],
                points=[x[i] for x in points],
                img_meta=img_metas[i]
            )
            results.append(result)
        return results

    # per image
    def _get_bboxes_single(self,
                           bbox_preds,
                           cls_scores,
                           points,
                           img_meta):
        mlvl_bboxes, mlvl_scores = [], []
        for bbox_pred, cls_score, point in zip(
            bbox_preds, cls_scores, points
        ):
            scores = cls_score.sigmoid()
            max_scores, _ = scores.max(dim=1)

            if len(scores) > self.test_cfg.nms_pre > 0:
                _, ids = max_scores.topk(self.test_cfg.nms_pre)
                bbox_pred = bbox_pred[ids]
                scores = scores[ids]
                point = point[ids]

            bboxes = self._bbox_pred_to_result(point, bbox_pred)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)

        bboxes = torch.cat(mlvl_bboxes)
        scores = torch.cat(mlvl_scores)
        bboxes, scores, labels = self._nms(bboxes, scores, img_meta)
        return bboxes, scores, labels

    # per scale
    def forward_single(self, x, reg_convs, cls_convs, reg_conv, cls_conv):
        raise NotImplementedError

    def _bbox_pred_to_loss(self, points, bbox_preds):
        raise NotImplementedError

    def _bbox_pred_to_result(self, points, bbox_preds):
        raise NotImplementedError

    def _nms(self, bboxes, scores, img_meta):
        raise NotImplementedError


@HEADS.register_module()
class ScanNetSparseYolo3DHead(SparseYolo3DHead):
    def forward_single(self, x, reg_convs, cls_convs, reg_conv, cls_conv):
        cls = cls_convs(x)
        reg = reg_convs(x)
        bbox_pred = torch.exp(reg_conv(reg).features)
        cls_score = cls_conv(cls).features

        bbox_preds, cls_scores, points = [], [], []
        for permutation in x.decomposition_permutations:
            bbox_preds.append(bbox_pred[permutation])
            cls_scores.append(cls_score[permutation])

        points = x.decomposed_coordinates
        for i in range(len(points)):
            points[i] = points[i] * self.voxel_size

        return bbox_preds, cls_scores, points

    def _bbox_pred_to_loss(self, points, bbox_preds):
        return aligned_bbox_pred_to_bbox(points, bbox_preds)

    def _bbox_pred_to_result(self, points, bbox_preds):
        return aligned_bbox_pred_to_bbox(points, bbox_preds)

    def _nms(self, bboxes, scores, img_meta):
        scores, labels = scores.max(dim=1)
        ids = scores > self.test_cfg.score_thr
        bboxes = bboxes[ids]
        scores = scores[ids]
        labels = labels[ids]
        ids = aligned_3d_nms(bboxes, scores, labels, self.test_cfg.iou_thr)
        bboxes = bboxes[ids]
        bboxes = torch.stack((
            (bboxes[:, 0] + bboxes[:, 3]) / 2.,
            (bboxes[:, 1] + bboxes[:, 4]) / 2.,
            (bboxes[:, 2] + bboxes[:, 5]) / 2.,
            bboxes[:, 3] - bboxes[:, 0],
            bboxes[:, 4] - bboxes[:, 1],
            bboxes[:, 5] - bboxes[:, 2]
        ), dim=1)
        bboxes = img_meta['box_type_3d'](bboxes, origin=(.5, .5, .5), box_dim=6, with_yaw=False)
        return bboxes, scores[ids], labels[ids]


def aligned_bbox_pred_to_bbox(points, bbox_pred):
    return torch.stack([
        points[:, 0] - bbox_pred[:, 0],
        points[:, 1] - bbox_pred[:, 2],
        points[:, 2] - bbox_pred[:, 4],
        points[:, 0] + bbox_pred[:, 1],
        points[:, 1] + bbox_pred[:, 3],
        points[:, 2] + bbox_pred[:, 5]
    ], -1)


def compute_centerness(bbox_targets):
    x_dims = bbox_targets[..., [0, 1]]
    y_dims = bbox_targets[..., [2, 3]]
    z_dims = bbox_targets[..., [4, 5]]
    centerness_targets = x_dims.min(dim=-1)[0] / x_dims.max(dim=-1)[0] * \
                         y_dims.min(dim=-1)[0] / y_dims.max(dim=-1)[0] * \
                         z_dims.min(dim=-1)[0] / z_dims.max(dim=-1)[0]
    # todo: sqrt ?
    return torch.sqrt(centerness_targets)


@BBOX_ASSIGNERS.register_module()
class ScanNetYolo3dAssigner(BaseAssigner):
    def __init__(self, regress_ranges, topk):
        self.regress_ranges = regress_ranges
        self.n_scales = len(regress_ranges)
        self.topk = topk

    def assign(self, points, gt_bboxes, gt_labels):
        float_max = 1e8
        assert len(points) == len(self.regress_ranges)
        # expand regress ranges to align with points
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i]).expand(len(points[i]), 2)
            for i in range(len(points))
        ]
        # concat all levels points and regress ranges
        regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        points = torch.cat(points, dim=0)

        # below is based on FCOSHead._get_target_single
        n_points = len(points)
        n_boxes = len(gt_bboxes)
        volumes = gt_bboxes.volume.to(points.device)
        volumes = volumes.expand(n_points, n_boxes).contiguous()
        regress_ranges = regress_ranges[:, None, :].expand(n_points, n_boxes, 2)
        gt_bboxes = torch.cat((gt_bboxes.gravity_center, gt_bboxes.dims), dim=1)
        gt_bboxes = gt_bboxes.to(points.device).expand(n_points, n_boxes, 6)
        xs, ys, zs = points[:, 0], points[:, 1], points[:, 2]
        xs = xs[:, None].expand(n_points, n_boxes)
        ys = ys[:, None].expand(n_points, n_boxes)
        zs = zs[:, None].expand(n_points, n_boxes)

        dx_min = xs - gt_bboxes[..., 0] + gt_bboxes[..., 3] / 2
        dx_max = gt_bboxes[..., 0] + gt_bboxes[..., 3] / 2 - xs
        dy_min = ys - gt_bboxes[..., 1] + gt_bboxes[..., 4] / 2
        dy_max = gt_bboxes[..., 1] + gt_bboxes[..., 4] / 2 - ys
        dz_min = zs - gt_bboxes[..., 2] + gt_bboxes[..., 5] / 2
        dz_max = gt_bboxes[..., 2] + gt_bboxes[..., 5] / 2 - zs
        bbox_targets = torch.stack((dx_min, dx_max, dy_min, dy_max, dz_min, dz_max), dim=-1)

        # condition1: inside a gt bbox
        inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0

        # condition2: limit the regression range for each location
        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (
                (max_regress_distance >= regress_ranges[..., 0])
                & (max_regress_distance <= regress_ranges[..., 1]))

        # condition3: limit topk locations per box by centerness
        centerness = compute_centerness(bbox_targets)
        centerness = torch.where(inside_gt_bbox_mask, centerness, torch.ones_like(centerness) * -1)
        centerness = torch.where(inside_regress_range, centerness, torch.ones_like(centerness) * -1)
        top_centerness = torch.topk(centerness, self.topk, dim=0).values[-1]
        inside_top_centerness = centerness > top_centerness.unsqueeze(0)


        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        volumes = torch.where(inside_gt_bbox_mask, volumes, torch.ones_like(volumes) * float_max)
        volumes = torch.where(inside_regress_range, volumes, torch.ones_like(volumes) * float_max)
        volumes = torch.where(inside_top_centerness, volumes, torch.ones_like(volumes) * float_max)
        min_area, min_area_inds = volumes.min(dim=1)

        labels = gt_labels[min_area_inds]
        labels = torch.where(min_area == float_max, torch.ones_like(labels) * -1, labels)
        bbox_targets = bbox_targets[range(n_points), min_area_inds]

        return bbox_targets, labels


@HEADS.register_module()
class SunRgbdSparseYolo3DHead(SparseYolo3DHead):
    def forward_single(self, x, reg_convs, cls_convs, reg_conv, cls_conv):
        cls = cls_convs(x)
        reg = reg_convs(x)
        reg_final = reg_conv(reg).features
        reg_distance = torch.exp(reg_final[:, :6])
        reg_angle = reg_final[:, 6:]
        bbox_pred = torch.cat((reg_distance, reg_angle), dim=1)
        cls_score = cls_conv(cls).features

        bbox_preds, cls_scores, points = [], [], []
        for permutation in x.decomposition_permutations:
            bbox_preds.append(bbox_pred[permutation])
            cls_scores.append(cls_score[permutation])

        points = x.decomposed_coordinates
        for i in range(len(points)):
            points[i] = points[i] * self.voxel_size

        return bbox_preds, cls_scores, points

    def _bbox_pred_to_loss(self, points, bbox_preds):
        return self._bbox_pred_to_bbox(points, bbox_preds)

    def _bbox_pred_to_result(self, points, bbox_preds):
        return self._bbox_pred_to_bbox(points, bbox_preds)

    @staticmethod
    def _bbox_pred_to_bbox(points, bbox_pred):
        if bbox_pred.shape[0] == 0:
            return bbox_pred

        shift = torch.stack((
            (bbox_pred[:, 1] - bbox_pred[:, 0]) / 2,
            (bbox_pred[:, 3] - bbox_pred[:, 2]) / 2,
            (bbox_pred[:, 5] - bbox_pred[:, 4]) / 2
        ), dim=-1).view(-1, 1, 3)
        shift = rotation_3d_in_axis(shift, bbox_pred[:, 6], axis=2)[:, 0, :]
        center = points + shift
        size = torch.stack((
            bbox_pred[:, 0] + bbox_pred[:, 1],
            bbox_pred[:, 2] + bbox_pred[:, 3],
            bbox_pred[:, 4] + bbox_pred[:, 5]
        ), dim=-1)
        return torch.cat((center, size, bbox_pred[:, 6:7]), dim=-1)

    def _nms(self, bboxes, scores, img_meta):
        # Add a dummy background class to the end. Nms needs to be fixed in the future.
        padding = scores.new_zeros(scores.shape[0], 1)
        scores = torch.cat([scores, padding], dim=1)
        bboxes_for_nms = torch.stack((
            bboxes[:, 0] - bboxes[:, 3] / 2,
            bboxes[:, 1] - bboxes[:, 4] / 2,
            bboxes[:, 0] + bboxes[:, 3] / 2,
            bboxes[:, 1] + bboxes[:, 4] / 2,
            bboxes[:, 6]
        ), dim=1)
        bboxes, scores, labels = box3d_multiclass_nms(
            mlvl_bboxes=bboxes,
            mlvl_bboxes_for_nms=bboxes_for_nms,
            mlvl_scores=scores,
            score_thr=self.test_cfg.score_thr,
            max_num=self.test_cfg.nms_pre,
            cfg=self.test_cfg
        )
        bboxes = img_meta['box_type_3d'](bboxes, origin=(.5, .5, .5))
        return bboxes, scores, labels


@BBOX_ASSIGNERS.register_module()
class SunRgbdYolo3dAssigner(BaseAssigner):
    def __init__(self, regress_ranges, topk):
        self.regress_ranges = regress_ranges
        self.n_scales = len(regress_ranges)
        self.topk = topk

    def assign(self, points, gt_bboxes, gt_labels):
        float_max = 1e8
        assert len(points) == len(self.regress_ranges)
        # expand regress ranges to align with points
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i]).expand(len(points[i]), 2)
            for i in range(len(points))
        ]
        # concat all levels points and regress ranges
        regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        points = torch.cat(points, dim=0)

        # below is based on FCOSHead._get_target_single
        n_points = len(points)
        n_boxes = len(gt_bboxes)
        volumes = gt_bboxes.volume.to(points.device)
        volumes = volumes.expand(n_points, n_boxes).contiguous()
        regress_ranges = regress_ranges[:, None, :].expand(n_points, n_boxes, 2)
        gt_bboxes = torch.cat((gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]), dim=1)
        gt_bboxes = gt_bboxes.to(points.device).expand(n_points, n_boxes, 7)
        expanded_points = points.unsqueeze(1).expand(n_points, n_boxes, 3)
        shift = torch.stack((
            expanded_points[..., 0] - gt_bboxes[..., 0],
            expanded_points[..., 1] - gt_bboxes[..., 1],
            expanded_points[..., 2] - gt_bboxes[..., 2]
        ), dim=-1).permute(1, 0, 2)
        shift = rotation_3d_in_axis(shift, -gt_bboxes[0, :, 6], axis=2).permute(1, 0, 2)
        centers = gt_bboxes[..., :3] + shift
        dx_min = centers[..., 0] - gt_bboxes[..., 0] + gt_bboxes[..., 3] / 2
        dx_max = gt_bboxes[..., 0] + gt_bboxes[..., 3] / 2 - centers[..., 0]
        dy_min = centers[..., 1] - gt_bboxes[..., 1] + gt_bboxes[..., 4] / 2
        dy_max = gt_bboxes[..., 1] + gt_bboxes[..., 4] / 2 - centers[..., 1]
        dz_min = centers[..., 2] - gt_bboxes[..., 2] + gt_bboxes[..., 5] / 2
        dz_max = gt_bboxes[..., 2] + gt_bboxes[..., 5] / 2 - centers[..., 2]
        bbox_targets = torch.stack((dx_min, dx_max, dy_min, dy_max, dz_min, dz_max, gt_bboxes[..., 6]), dim=-1)

        # condition1: inside a gt bbox
        inside_gt_bbox_mask = bbox_targets[..., :6].min(-1)[0] > 0  # skip angle

        # condition2: limit the regression range for each location
        max_regress_distance = bbox_targets[..., :6].max(-1)[0]  # skip angle
        inside_regress_range = (
                (max_regress_distance >= regress_ranges[..., 0])
                & (max_regress_distance <= regress_ranges[..., 1]))

        # condition3: limit topk locations per box by centerness
        centerness = compute_centerness(bbox_targets)
        centerness = torch.where(inside_gt_bbox_mask, centerness, torch.ones_like(centerness) * -1)
        centerness = torch.where(inside_regress_range, centerness, torch.ones_like(centerness) * -1)
        top_centerness = torch.topk(centerness, self.topk, dim=0).values[-1]
        inside_top_centerness = centerness > top_centerness.unsqueeze(0)

        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        volumes = torch.where(inside_gt_bbox_mask, volumes, torch.ones_like(volumes) * float_max)
        volumes = torch.where(inside_regress_range, volumes, torch.ones_like(volumes) * float_max)
        volumes = torch.where(inside_top_centerness, volumes, torch.ones_like(volumes) * float_max)
        min_area, min_area_inds = volumes.min(dim=1)

        labels = gt_labels[min_area_inds]
        labels = torch.where(min_area == float_max, torch.ones_like(labels) * -1, labels)
        bbox_targets = bbox_targets[range(n_points), min_area_inds]

        return bbox_targets, labels