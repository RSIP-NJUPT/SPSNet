# Copyright (c) OpenMMLab. All rights reserved.
"""This script is in the experimental verification stage and cannot be
guaranteed to be completely correct. Currently Grad-based CAM and Grad-free CAM
are supported.

The target detection task is different from the classification task. It not
only includes the AM map of the category, but also includes information such as
bbox and mask, so this script is named bboxam.
"""

import argparse
import os.path
import warnings
from functools import partial

import cv2
import mmcv
from mmengine import Config, DictAction, MessageHub
from mmengine.utils import ProgressBar

try:
    from pytorch_grad_cam import AblationCAM, EigenCAM
except ImportError:
    raise ImportError('Please run `pip install "grad-cam"` to install '
                      'pytorch_grad_cam package.')

from mmyolo.utils.boxam_utils import (BoxAMDetectorVisualizer,
                                      BoxAMDetectorWrapper, DetAblationLayer,
                                      DetBoxScoreTarget, GradCAM,
                                      GradCAMPlusPlus, reshape_transform)
from mmyolo.utils.misc import get_file_list
import torch
import numpy as np
from pytorch_grad_cam.utils.image import scale_cam_image, show_cam_on_image
from mmengine.structures import InstanceData
import math


class DetectionBoxAMDetectorWrapper(BoxAMDetectorWrapper):
    def __call__(self, *args, **kwargs):
        assert self.input_data is not None
        if self.is_need_loss:
            # Maybe this is a direction that can be optimized
            # self.detector.init_weights()

            # self.detector.bbox_head.head_module.training = True
            if hasattr(self.detector.bbox_head, 'featmap_sizes'):
                # Prevent the model algorithm error when calculating loss
                self.detector.bbox_head.featmap_sizes = None

            data_ = {}
            data_['inputs'] = [self.input_data['inputs']]
            data_['data_samples'] = [self.input_data['data_samples']]
            data = self.detector.data_preprocessor(data_, training=False)
            loss = self.detector._run_forward(data, mode='loss')

            if hasattr(self.detector.bbox_head, 'featmap_sizes'):
                self.detector.bbox_head.featmap_sizes = None

            return [loss]
        else:
            # self.detector.bbox_head.head_module.training = False
            with torch.no_grad():
                results = self.detector.test_step(self.input_data)
                return results


GRAD_FREE_METHOD_MAP = {
    'ablationcam': AblationCAM,
    'eigencam': EigenCAM,
    # 'scorecam': ScoreCAM, # consumes too much memory
}

GRAD_BASED_METHOD_MAP = {'gradcam': GradCAM, 'gradcam++': GradCAMPlusPlus}

ALL_SUPPORT_METHODS = list(GRAD_FREE_METHOD_MAP.keys()
                           | GRAD_BASED_METHOD_MAP.keys())

IGNORE_LOSS_PARAMS = {
    'yolov5': ['loss_obj'],
    'yolov6': ['loss_cls'],
    'yolox': ['loss_obj'],
    'rtmdet': ['loss_cls'],
    'yolov7': ['loss_obj'],
    'yolov8': ['loss_cls'],
    'ppyoloe': ['loss_cls'],
}

# This parameter is required in some algorithms
# for calculating Loss
message_hub = MessageHub.get_current_instance()
message_hub.runtime_info['epoch'] = 0


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize Box AM')
    parser.add_argument(
        'img', help='Image path, include image file, dir and URL.')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--method',
        default='gradcam',
        choices=ALL_SUPPORT_METHODS,
        help='Type of method to use, supports '
             f'{", ".join(ALL_SUPPORT_METHODS)}.')
    parser.add_argument(
        '--target-layers',
        default=['neck.out_layers2'],
        nargs='+',
        type=str,
        help='The target layers to get Box AM, if not set, the tool will '
             'specify the neck.out_layers[2]')
    parser.add_argument(
        '--out-dir', default='./output', help='Path to output file')
    parser.add_argument(
        '--show', action='store_true', help='Show the CAM results')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.25, help='Bbox score threshold')
    parser.add_argument(
        '--topk',
        type=int,
        default=-1,
        help='Select topk predict resutls to show. -1 are mean all.')
    parser.add_argument(
        '--max-shape',
        nargs='+',
        type=int,
        default=-1,
        help='max shapes. Its purpose is to save GPU memory. '
             'The activation map is scaled and then evallateral_convsuated. '
             'If set to -1, it means no scaling.')
    parser.add_argument(
        '--preview-model',
        default=False,
        action='store_true',
        help='To preview all the model layers')
    parser.add_argument(
        '--norm-in-bbox', action='store_true', help='Norm in bbox of am image')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value to '
             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
             'Note that the quotation marks are necessary and that no white space '
             'is allowed.')
    # Only used by AblationCAM
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='batch of inference of AblationCAM')
    parser.add_argument(
        '--ratio-channels-to-ablate',
        type=int,
        default=0.5,
        help='Making it much faster of AblationCAM. '
             'The parameter controls how many channels should be ablated')

    args = parser.parse_args()
    return args


def init_detector_and_visualizer(args, cfg):
    max_shape = args.max_shape
    if not isinstance(max_shape, list):
        max_shape = [args.max_shape]
    assert len(max_shape) == 1 or len(max_shape) == 2

    model_wrapper = DetectionBoxAMDetectorWrapper(
        cfg, args.checkpoint, args.score_thr, device=args.device)

    if args.preview_model:
        print(model_wrapper.detector)
        print('\n Please remove `--preview-model` to get the BoxAM.')
        return None, None

    target_layers = []
    for target_layer in args.target_layers:
        try:
            target_layers.append(
                eval(f'model_wrapper.detector.{target_layer}'))
        except Exception as e:
            print(model_wrapper.detector)
            raise RuntimeError('layer does not exist', e)

    ablationcam_extra_params = {
        'batch_size': args.batch_size,
        'ablation_layer': DetAblationLayer(),
        'ratio_channels_to_ablate': args.ratio_channels_to_ablate
    }

    if args.method in GRAD_BASED_METHOD_MAP:
        method_class = GRAD_BASED_METHOD_MAP[args.method]
        is_need_grad = True
    else:
        method_class = GRAD_FREE_METHOD_MAP[args.method]
        is_need_grad = False

    boxam_detector_visualizer = BoxAMDetectorVisualizer(
        method_class,
        model_wrapper,
        target_layers,
        reshape_transform=partial(
            reshape_transform, max_shape=max_shape, is_need_grad=is_need_grad),
        is_need_grad=is_need_grad,
        extra_params=ablationcam_extra_params)
    return model_wrapper, boxam_detector_visualizer


def show_am(
            image: np.ndarray,
            pred_instance: InstanceData,
            grayscale_am: np.ndarray,
            with_norm_in_bboxes: bool = False):
    """Normalize the AM to be in the range [0, 1] inside every bounding
        boxes, and zero outside of the bounding boxes."""

    boxes = pred_instance.bboxes
    labels = pred_instance.labels

    if with_norm_in_bboxes is True:
        boxes = boxes.astype(np.int32)
        renormalized_am = np.zeros(grayscale_am.shape, dtype=np.float32)
        images = []
        # for cx,cy,h,w,angle in boxes:
        #     # RSDD数据集标注框初始是竖着的
        #     # angle += 1 / 2 * math.pi
        #
        #     # 将robndbox信息格式化为8个点表示的旋转目标框
        #     cos_a = math.cos(angle)
        #     sin_a = math.sin(angle)
        #
        #     Ax = cx - 0.5 * w * cos_a - 0.5 * h * sin_a
        #     Ay = cy - 0.5 * w * sin_a + 0.5 * h * cos_a
        #
        #     Bx = cx + 0.5 * w * cos_a - 0.5 * h * sin_a
        #     By = cy + 0.5 * w * sin_a + 0.5 * h * cos_a
        #
        #     Cx = cx + 0.5 * w * cos_a + 0.5 * h * sin_a
        #     Cy = cy + 0.5 * w * sin_a - 0.5 * h * cos_a
        #
        #     Dx = cx - 0.5 * w * cos_a + 0.5 * h * sin_a
        #     Dy = cy - 0.5 * w * sin_a - 0.5 * h * cos_a
        #     min_x = min(Ax,Bx,Cx,Dx)
        #     if min_x < 0:
        #         min_x = 0
        #     min_y = min(Ay,By,Cy,Dy)
        #     if min_y < 0:
        #         min_y = 0
        #     max_x = max(Ax,Bx,Cx,Dx)
        #     if max_x > 1024:
        #         max_x = 1024
        #     max_y = max(Ay,By,Cy,Dy)
        #     if max_y > 1024:
        #         max_y = 1024
        #     img = renormalized_am * 0
        #     img[int(min_y):int(max_y), int(min_x):int(max_x)] = scale_cam_image(
        #         [grayscale_am[int(min_y):int(max_y), int(min_x):int(max_x)].copy()])[0]
        #     images.append(img)

        for x1, y1, x2, y2 in boxes:
            img = renormalized_am * 0
            img[y1:y2, x1:x2] = scale_cam_image(
                [grayscale_am[y1:y2, x1:x2].copy()])[0]
            images.append(img)

        renormalized_am = np.max(np.float32(images), axis=0)
        renormalized_am = scale_cam_image([renormalized_am])[0]
    else:
        renormalized_am = grayscale_am

    am_image_renormalized = show_cam_on_image(
        image / 255, renormalized_am, use_rgb=False)

    # image_with_bounding_boxes = self._draw_boxes(
    #     boxes, labels, am_image_renormalized, pred_instance.get('scores'))
    # return image_with_bounding_boxes
    return am_image_renormalized


def main():
    args = parse_args()

    # hard code
    ignore_loss_params = None
    for param_keys in IGNORE_LOSS_PARAMS:
        if param_keys in args.config:
            print(f'The algorithm currently used is {param_keys}')
            ignore_loss_params = IGNORE_LOSS_PARAMS[param_keys]
            break

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if not os.path.exists(args.out_dir) and not args.show:
        os.makedirs(args.out_dir)

    model_wrapper, boxam_detector_visualizer = init_detector_and_visualizer(
        args, cfg)

    # get file list
    image_list, source_type = get_file_list(args.img)

    progress_bar = ProgressBar(len(image_list))

    for image_path in image_list:
        image = cv2.imread(image_path)
        model_wrapper.set_input_data(image)

        # forward detection results
        result = model_wrapper()[0]

        pred_instances = result.pred_instances
        # Get candidate predict info with score threshold
        pred_instances = pred_instances[pred_instances.scores > args.score_thr]

        if len(pred_instances) == 0:
            warnings.warn('empty detection results! skip this')
            continue

        if args.topk > 0:
            pred_instances = pred_instances[:args.topk]

        targets = [
            DetBoxScoreTarget(
                pred_instances,
                device=args.device,
                ignore_loss_params=ignore_loss_params)
        ]

        if args.method in GRAD_BASED_METHOD_MAP:
            model_wrapper.need_loss(True)
            model_wrapper.set_input_data(image, pred_instances)
            boxam_detector_visualizer.switch_activations_and_grads(
                model_wrapper)

        # get box am image
        grayscale_boxam = boxam_detector_visualizer(image, targets=targets)

        # draw cam on image
        pred_instances = pred_instances.numpy()
        image_with_bounding_boxes = show_am(
            image,
            pred_instances,
            grayscale_boxam,
            with_norm_in_bboxes=args.norm_in_bbox)

        if source_type['is_dir']:
            filename = os.path.relpath(image_path, args.img).replace('/', '_')
        else:
            filename = os.path.basename(image_path)
        out_file = None if args.show else os.path.join(args.out_dir, filename)

        if out_file:
            mmcv.imwrite(image_with_bounding_boxes, out_file)
        else:
            cv2.namedWindow(filename, 0)
            cv2.imshow(filename, image_with_bounding_boxes)
            cv2.waitKey(0)

        # switch
        if args.method in GRAD_BASED_METHOD_MAP:
            model_wrapper.need_loss(False)
            boxam_detector_visualizer.switch_activations_and_grads(
                model_wrapper)

        progress_bar.update()

    if not args.show:
        print(f'All done!'
              f'\nResults have been saved at {os.path.abspath(args.out_dir)}')


if __name__ == '__main__':
    main()
