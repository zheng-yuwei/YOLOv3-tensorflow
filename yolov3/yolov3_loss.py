# -*- coding: utf-8 -*-
"""
File yolov3_loss.py
@author:ZhengYuwei
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from yolov3.yolov3_decoder import YOLOv3Decoder
from yolov3.label_decoder import LabelDecoder


class YOLOv3Loss(object):
    """ YOLOv3损失函数 """
    
    def __init__(self, head_grid_sizes, class_num, anchor_boxes, iou_thresh, loss_weights,
                 rectified_coord_num=0, rectified_loss_weight=None,
                 is_focal_loss=False, focal_alpha=0.25, focal_gamma=2.0, is_tiou_recall=False):
        """
        YOLO v3 损失函数参数的初始化
        :param head_grid_sizes: YOLO v3输出head的所有尺度列表
        :param class_num: 类别数
        :param anchor_boxes: YOLO v3的预定义anchors
        :param iou_thresh: 任一grid、anchor预测的bounding box与所有实际目标的IOU，小于该阈值且不为最大IOU则为background，
                            需要计算noobj情况下的IOU损失；大于该阈值且不为最大IOU则不计算损失
        :param loss_weights: 不同损失项的权重，[coord_xy, coord_wh, noobj, obj, cls_prob]
        :param rectified_coord_num: 前期给坐标做矫正损失的图片数
        :param rectified_loss_weight: 前期矫正坐标的损失的权重，None则表示默认值[0.01, 0.01, 0.01]
        :param is_focal_loss: 是否针对前/背景IOU损失函数使用focal loss
        :param focal_alpha: the same as wighting factor in balanced cross entropy, default 0.25
        :param focal_gamma: focusing parameter for modulating factor (1-p), default 2.0
        :param is_tiou_recall: 是否使用TIOU-recall替换一般的IOU计算
        """
        # YOLO v3输出解码器和标签解码器
        self.predict_decoder = YOLOv3Decoder(head_grid_sizes, class_num, anchor_boxes)
        self.target_decoder = LabelDecoder(head_grid_sizes)
        # 每个head的grid尺寸
        head_8_grid_size, head_16_grid_size, head_32_grid_size = head_grid_sizes
        self.head_8_height, self.head_8_width = head_8_grid_size
        self.head_16_height, self.head_16_width = head_16_grid_size
        self.head_32_height, self.head_32_width = head_32_grid_size
        self.head_8_wh = tf.constant([self.head_8_width, self.head_8_height], dtype=tf.float32)
        self.head_16_wh = tf.constant([self.head_16_width, self.head_16_height], dtype=tf.float32)
        self.head_32_wh = tf.constant([self.head_32_width, self.head_32_height], dtype=tf.float32)
        # 坐标损失项、背景IOU损失项权重
        self.coord_xy_weight, self.coord_wh_weight, self.noobj_weight, self.obj_weight, self.cls_weight = \
            [tf.constant(weight, dtype=tf.float32) for weight in np.transpose(loss_weights)]
        # bounding box & anchor_boxes
        self.head_8_anchor_boxes, self.head_16_anchor_boxes, self.head_32_anchor_boxes = anchor_boxes
        self.head_8_box_num, self.head_16_box_num, self.head_32_box_num = [len(boxes) for boxes in anchor_boxes]
        self.coord_num = 4
        self.conf_num = 1
        self.class_num = class_num
        self.box_len = self.coord_num + self.conf_num + self.class_num
        self.iou_thresh = iou_thresh
        # 损失函数变种选项
        self.is_focal_loss = is_focal_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.is_tiou_recall = is_tiou_recall
        # 前期训练 rectified loss
        self.rectified_coord_num = rectified_coord_num  # 预定义的坐标校正图片数
        if rectified_loss_weight is None:
            rectified_loss_weight = [0.01, 0.01, 0.01]
        elif len(rectified_loss_weight) != 3:
            raise ValueError('rectified_loss_weight长度必须为3，表示head 8, 16, 32的前期校正损失项权重')
        self.rectified_8_weight, self.rectified_16_weight, self.rectified_32_weight = rectified_loss_weight
        # 前期矫正的图片数
        self.current_num = keras.backend.variable(0, dtype=tf.int32, name='current_image_num')
        # 用于记录损失函数的细节
        with tf.variable_scope('loss_detail'):
            self.rectified_coord_loss = tf.get_variable('rectified_coord_loss', shape=(3, ),
                                                        initializer=tf.constant_initializer(0.0))
            self.coord_loss_xy = tf.get_variable('coord_loss_xy', shape=(3, ), initializer=tf.constant_initializer(0.0))
            self.coord_loss_wh = tf.get_variable('coord_loss_wh', shape=(3, ), initializer=tf.constant_initializer(0.0))
            self.noobj_iou_loss = tf.get_variable('noobj_iou_loss', shape=(3, ),
                                                  initializer=tf.constant_initializer(0.0))
            self.obj_iou_loss = tf.get_variable('obj_iou_loss', shape=(3, ), initializer=tf.constant_initializer(0.0))
            self.class_loss = tf.get_variable('class_loss', shape=(3, ), initializer=tf.constant_initializer(0.0))
        
    def loss(self, targets, predicts):
        """
        计算YOLO v3损失函数
        :param targets: 尺寸为(N, obj_num * 5)，每一行为：obj_num * [x, y, w, h, class],
                        (x, y): 归一化中心点坐标
                        (w, h): 归一化宽高
                        class: 目标所属类别标签
                        obj_num: 单张图中目标个数
        :param predicts: YOLO v3网络输出的预测，3个head组成的矩阵
                        (N, H/32, W/32, head_32_channel*16 + head_32_channel*4 + head_32_channel)矩阵
        :return: 总损失
        """
        # 0. 解码网络的输出，拆分3个head，每个head分别解码：
        # 解码前的坐标预测值[t_x, t_y, t_w, t_h], 维度为(N, H, W, B, 4)
        # 预测值[center_x, center_y, w, h, score, class_prob], 维度为(N, H, W, B, 4 + 1 + class_num)
        # 以及解码后的bounding prediction [left_top_x, left_top_y, right_bottom_x, right_bottom_y]，(N, H, W, B, 4)
        [(_head_8_predicts, head_8_predicts, head_8_predicts_boxes),
         (_head_16_predicts, head_16_predicts, head_16_predicts_boxes),
         (_head_32_predicts, head_32_predicts, head_32_predicts_boxes)] = self.predict_decoder.decode(predicts)
        # 1. 解码标签：(N, obj_num, 5)的标签矩阵, (N, obj_num, 4)的bounding boxes坐标
        [(head_8_targets, head_8_targets_boxes),
         (head_16_targets, head_16_targets_boxes),
         (head_32_targets, head_32_targets_boxes)] = self.target_decoder.decode(targets)
        # 2. 逐张图片计算损失函数，按样本维度遍历，得到损失矩阵 (5, 3)，3个head，顺序为 /8, /16, /32，每个head 5项损失
        predicts_targets = (head_8_predicts, head_8_predicts_boxes,
                            head_16_predicts, head_16_predicts_boxes,
                            head_32_predicts, head_32_predicts_boxes,
                            head_8_targets, head_8_targets_boxes,
                            head_16_targets, head_16_targets_boxes,
                            head_32_targets, head_32_targets_boxes)
        yolov3_loss = tf.map_fn(self._single_image_loss, predicts_targets, dtype=tf.float32, parallel_iterations=1)
        yolov3_loss = tf.reduce_mean(yolov3_loss, axis=0)
        # 3. 汇总并记录所有损失 (5, 3) + (1, 3) -> (6, 3)
        # 用于记录损失函数的细节，在logger_callback里使用
        update_op = [
            self.coord_loss_xy.assign(yolov3_loss[0]),
            self.coord_loss_wh.assign(yolov3_loss[1]),
            self.noobj_iou_loss.assign(yolov3_loss[2]),
            self.obj_iou_loss.assign(yolov3_loss[3]),
            self.class_loss.assign(yolov3_loss[4]),
        ]
        # 4. 前期矫正的图片数小于预定义的坐标校正图片数，则继续加坐标校正损失
        # [rectified_coord_loss, coord_loss_xy, coord_loss_wh, noobj_iou_loss, obj_iou_loss, class_loss]
        # [rectified_8_loss, rectified_16_loss, rectified_32_loss]
        total_loss = tf.cond(self.current_num <= self.rectified_coord_num,
                             lambda:  tf.concat([yolov3_loss,
                                                 self._get_rectified_coord_loss(_head_8_predicts,
                                                                                _head_16_predicts,
                                                                                _head_32_predicts)], axis=0),
                             lambda: yolov3_loss)
        update_op = tf.cond(self.current_num <= self.rectified_coord_num,
                            lambda: update_op + [self.rectified_coord_loss.assign(total_loss[5])],
                            lambda: update_op + [self.rectified_coord_loss.assign(tf.zeros(shape=(3, ),
                                                                                           dtype=tf.float32))])
        # 5. 汇总所有损失
        with tf.control_dependencies(update_op):
            total_loss = tf.reduce_sum(total_loss)
        return total_loss

    def _get_rectified_coord_loss(self, _head_8_predicts, _head_16_predicts, _head_32_predicts):
        """
        YOLO源码技巧：train-from-scratch 前期（12800 pic）坐标回归时，将预测wh回归为anchor，预测xy回归为grid的中心点：
        w_{anchor} * exp(t_{wh})= w_{anchor} => t_{wh} = 0
        left_top + sigmoid(t_{xy}) = left_top + 0.5 => t_{xy} = 0
        这样的好处是，有一个好的初始预测框，并且loss的梯度在最大处
        :param _head_8_predicts: YOLO v3输出的stride 8坐标预测值[t_x, t_y, t_w, t_h], 维度为(N, H, W, B, 4)
        :param _head_16_predicts: YOLO v3输出的stride 16坐标预测值[t_x, t_y, t_w, t_h], 维度为(N, H, W, B, 4)
        :param _head_32_predicts: YOLO v3输出的stride 32坐标预测值[t_x, t_y, t_w, t_h], 维度为(N, H, W, B, 4)
        :return: 校正损失
        """
        sample_num = tf.shape(_head_8_predicts)[0]
        with tf.control_dependencies([keras.backend.update_add(self.current_num, sample_num)]):
            head_8_rectified_loss = tf.reduce_sum(tf.square(_head_8_predicts), axis=[1, 2, 3, 4])
            head_16_rectified_loss = tf.reduce_sum(tf.square(_head_16_predicts), axis=[1, 2, 3, 4])
            head_32_rectified_loss = tf.reduce_sum(tf.square(_head_32_predicts), axis=[1, 2, 3, 4])
    
            rectified_8_loss = tf.multiply(self.rectified_8_weight,
                                           tf.reduce_mean(head_8_rectified_loss, keepdims=True))
            rectified_16_loss = tf.multiply(self.rectified_16_weight,
                                            tf.reduce_mean(head_16_rectified_loss, keepdims=True))
            rectified_32_loss = tf.multiply(self.rectified_32_weight,
                                            tf.reduce_mean(head_32_rectified_loss, keepdims=True))
            rectified_loss = tf.stack([rectified_8_loss, rectified_16_loss, rectified_32_loss], axis=-1)
        return rectified_loss

    def _single_image_loss(self, predict_target):
        """
        单张图片的损失函数：取出有效label，计算IOU，计算loss
        :param predict_target: 包含3个head的预测和标签数据
                head_x_target: head x的标签，[[enter_x, center_y, width, height, class] ...]，(obj_num, 5)
                head_x_target_boxes: [[left_top_x, left_top_y, right_bottom_x, right_bottom_y] ...]，(obj_num, 4)
                head_x_predict: head x的预测，(H, W, B, 4 + 1 + class_num)
                head_x_predict_boxes: (H, W, B, 4), [[left_top_x, left_top_y, right_bottom_x, right_bottom_y]...]
        :return: 该图片的损失，(5, 3), 3个head，顺序为 /8, /16, /32
                [3 * [coord_loss_xy, coord_loss_wh, noobj_iou_loss, obj_iou_loss, class_loss]]
        """
        # 0. 解析输入
        (head_8_predict, head_8_predict_boxes,
         head_16_predict, head_16_predict_boxes,
         head_32_predict, head_32_predict_boxes,
         head_8_target, head_8_target_boxes,
         head_16_target, head_16_target_boxes,
         head_32_target, head_32_target_boxes) = predict_target
        # 1. 取出有效目标label：(obj_num, 5), (obj_num, 4) => (valid_num, 5), (valid_num, 4)
        (head_8_target, head_8_target_boxes,
         head_16_target, head_16_target_boxes,
         head_32_target, head_32_target_boxes,
         valid_obj_num) = self._get_valid_target(head_8_target, head_8_target_boxes,
                                                 head_16_target, head_16_target_boxes,
                                                 head_32_target, head_32_target_boxes)
        # 2. 计算IOU及response anchor的位置
        with tf.name_scope('calculate_iou'):
            # 所有grid、所有anchor预测框和所有实际目标的IOU，统计各anchor的最大IOU = head_x_max_iou，(H, W, B)
            # 中心点grid所有anchor与对应的实际物体的最大IOU (valid_num,) = head_x_response_max_iou
            # 及其坐标[[H, W, B] * valid_num] = head_x_target_grid_xyz
            head_8_max_iou, head_8_response_max_iou, head_8_target_grid_xyz = self._calc_iou(
                head_8_target, head_8_target_boxes, head_8_predict, head_8_predict_boxes,  valid_obj_num)
            head_16_max_iou, head_16_response_max_iou, head_16_target_grid_xyz = self._calc_iou(
                head_16_target, head_16_target_boxes, head_16_predict, head_16_predict_boxes, valid_obj_num)
            head_32_max_iou, head_32_response_max_iou, head_32_target_grid_xyz = self._calc_iou(
                head_32_target, head_32_target_boxes, head_32_predict, head_32_predict_boxes, valid_obj_num)
            # 获取3个head中和gt最大的IOU和对应的坐标
            head_8_max_pos = tf.where(tf.logical_and(head_8_response_max_iou >= head_16_response_max_iou,
                                                     head_8_response_max_iou >= head_32_response_max_iou))
            head_16_max_pos = tf.where(tf.logical_and(head_16_response_max_iou >= head_8_response_max_iou,
                                                      head_16_response_max_iou >= head_32_response_max_iou))
            head_32_max_pos = tf.where(tf.logical_and(head_32_response_max_iou >= head_8_response_max_iou,
                                                      head_32_response_max_iou >= head_16_response_max_iou))
        # 3. 计算每个head的loss：(5,), [coord_loss_xy, coord_loss_wh, noobj_iou_loss, obj_iou_loss, class_loss]
        head_8_loss = self._single_head_loss(0, head_8_predict, head_8_target, head_8_max_iou, head_8_response_max_iou,
                                             head_8_target_grid_xyz, head_8_max_pos, self.head_8_height,
                                             self.head_8_width, self.head_8_box_num, self.iou_thresh)
        head_16_loss = self._single_head_loss(1, head_16_predict, head_16_target, head_16_max_iou,
                                              head_16_response_max_iou, head_16_target_grid_xyz, head_16_max_pos,
                                              self.head_16_height, self.head_16_width, self.head_16_box_num,
                                              self.iou_thresh)
        head_32_loss = self._single_head_loss(2, head_32_predict, head_32_target, head_32_max_iou,
                                              head_32_response_max_iou, head_32_target_grid_xyz, head_32_max_pos,
                                              self.head_32_height, self.head_32_width, self.head_32_box_num,
                                              self.iou_thresh)
        loss = tf.stack([head_8_loss, head_16_loss, head_32_loss], axis=-1)
        return loss
    
    @staticmethod
    def _get_valid_target(head_8_target, head_8_target_boxes,
                          head_16_target, head_16_target_boxes,
                          head_32_target, head_32_target_boxes):
        """
        根据标签的类别列，判断是否为padding的数据，以取出有效目标label
        :param head_8_target: stride 8的输出head的标签，(obj_num, 5),[[center_x, center_y, width, height, class]...]
        :param head_8_target_boxes: (obj_num, 4), [[left_top_x, left_top_y, right_bottom_x, right_bottom_y] ...]
        :param head_16_target: stride 16的输出head的标签，(obj_num, 5)
        :param head_16_target_boxes: (obj_num, 4)
        :param head_32_target: stride 32的输出head的标签，(obj_num, 5)
        :param head_32_target_boxes: (obj_num, 4)
        :return: 保留有效目标label，分别将target和target_boxes的obj_num，降为valid_obj_num
        """
        # 我在tf.data加入padding -1使得每张图的标签一样长，得以构造tf.date.Dataset数据，可查看dataset/file_util.py
        valid_obj_mask = head_8_target[:, 0] >= 0
        valid_obj_num = tf.count_nonzero(valid_obj_mask)
        valid_obj_indices = tf.where(valid_obj_mask)
        head_8_target = tf.gather_nd(head_8_target, valid_obj_indices)
        head_8_target_boxes = tf.gather_nd(head_8_target_boxes, valid_obj_indices)
        head_16_target = tf.gather_nd(head_16_target, valid_obj_indices)
        head_16_target_boxes = tf.gather_nd(head_16_target_boxes, valid_obj_indices)
        head_32_target = tf.gather_nd(head_32_target, valid_obj_indices)
        head_32_target_boxes = tf.gather_nd(head_32_target_boxes, valid_obj_indices)
        targets = (head_8_target, head_8_target_boxes,
                   head_16_target, head_16_target_boxes,
                   head_32_target, head_32_target_boxes,
                   valid_obj_num)
        return targets
    
    def _calc_iou(self, target, target_boxes, predict, predict_boxes, valid_obj_num):
        """
        所有grid的所有anchor的预测框，与所有真实物体框的最大IOU（后续用于确定是否是背景）
        物体中心点所在的grid的所有anchor的预测框，与对应的真实物体框的最大IOU及其位置（后续用于确定前景及loss计算）
        :param target: 有效物体label，(valid_num, 5)
        :param target_boxes: 有效物体bounding prediction，(valid_num, 4)
        :param predict: 预测label，(H, W, B, 4+1+class_num)
        :param predict_boxes: 预测bounding prediction，(H, W, B, 4)
        :param valid_obj_num: 有效物体label数
        :return: 预测框与所有gt物体框的最大IOU，(H, W, B)，
                 中心点grid的anchor与gt的最大IOU (valid_num,)及其位置[[H, W, B] * valid_num],(valid_num, 3)
        """
        # 1. 所有grid、所有anchor的预测框面积，取出真实物体中心点grid的所有anchor的预测框面积：A(D)
        predict_area = predict[:, :, :, 2] * predict[:, :, :, 3]  # (H, W, B)
        # response boxes的预测框面积：A(D)
        response_grid_xy = tf.cast(tf.floor(target[:, 0:2]), dtype=tf.int32)
        response_grid_xy = tf.reverse(response_grid_xy, axis=[-1])  # 由坐标获取下标：[center_x, center_y]=>(H, W)
        response_area = tf.gather_nd(predict_area, response_grid_xy)  # 获得response grid的预测输出， (valid_num, B)
        # 2. 真实框面积: A(G)
        target_area = target[:, 2] * target[:, 3]  # (valid_num,)
        # 3. 所有grid的所有anchor的预测框与真实框相交的面积，中心点grid的所有anchor预测框与对应gt框的相交面积: A(G and D)
        tile_predict_boxes = tf.tile(tf.expand_dims(predict_boxes, axis=-2), [1, 1, 1, valid_obj_num, 1])
        left_top = tf.maximum(tile_predict_boxes[:, :, :, :, 0:2], target_boxes[:, 0:2])
        right_bottom = tf.minimum(tile_predict_boxes[:, :, :, :, 2:4], target_boxes[:, 2:4])
        inter_wh = tf.maximum(right_bottom - left_top, 0)
        inter_area = inter_wh[:, :, :, :, 0] * inter_wh[:, :, :, :, 1]  # (H, W, B, valid_num)
        # response boxes的A(G and D)
        response_boxes = tf.gather_nd(predict_boxes, response_grid_xy)  # 获得response grid预测的boxes，(valid_num, B, 4)
        expand_target_boxes = tf.expand_dims(target_boxes, axis=1)
        response_left_top = tf.maximum(response_boxes[:, :, 0:2], expand_target_boxes[:, :, 0:2])
        response_right_bottom = tf.minimum(response_boxes[:, :, 2:4], expand_target_boxes[:, :, 2:4])
        response_inter_wh = tf.maximum(response_right_bottom - response_left_top, 0)
        response_inter_area = response_inter_wh[:, :, 0] * response_inter_wh[:, :, 1]  # (valid_num, B)
        # 4. 计算IOU
        # 所有grid的所有anchor与所有gt的IOU，gt中最大的IOU作为该anchor的IOU: A(G and D) / [A(D) + A(G) - A(G and D)]
        predict_area = tf.tile(tf.expand_dims(predict_area, axis=-1), [1, 1, 1, valid_obj_num])
        iou = inter_area / (predict_area + target_area - inter_area)
        if self.is_tiou_recall:
            # A(G and D) / [A(D) + A(G) - A(G and D)] * A(G and D) / A(G)
            iou = iou * inter_area / target_area
        max_iou = tf.reduce_max(iou, axis=-1, keepdims=False)  # (H, W, B)
        # 中心点grid所有anchor与对应的实际物体的最大IOU (valid_num,)，及其坐标(valid_num, 3),[[H, W, B] * valid_num]
        response_iou = response_inter_area / (response_area + tf.expand_dims(target_area, axis=-1) - response_inter_area)
        if self.is_tiou_recall:
            response_iou = response_iou * response_inter_area / tf.expand_dims(target_area, axis=-1)
        response_max_iou = tf.reduce_max(response_iou, axis=-1, keepdims=False)
        max_arg = tf.arg_max(response_iou, dimension=-1, output_type=tf.int32)
        response_grid_xyz = tf.concat([response_grid_xy, tf.expand_dims(max_arg, axis=-1)], axis=-1)
        return max_iou, response_max_iou, response_grid_xyz
    
    def _single_head_loss(self, head_index, predict, target, max_iou, response_max_iou, target_grid_xyz, max_pos,
                          height, width, box_num, iou_thresh):
        """
        计算指定head的损失
        :param head_index: 表示第几个head，主要用来作为损失项权重系数的下标，因为不同head的损失项系数不一样
        :param predict: 对应head的预测，[中心点、宽高、IOU、类别概率]，(H, W, B, 4 + 1 + class_num)
        :param target: 对应head的label，[[enter_x, center_y, width, height, class] ...]，(obj_num, 5)
        :param max_iou: 各预测anchor的最大IOU，(H, W, B)
        :param response_max_iou: 负责对应label目标的anchor的最大IOU，(valid_num,)
        :param target_grid_xyz: 负责目标（也就是grid中最大IOU）的anchor的位置, (valid_num, 3), [(H, W, B), ...]
        :param max_pos: 跨head最大IOU的标记
        :param height: 对应head的高
        :param width: 对应head的宽
        :param box_num: 对应head的anchor数
        :param iou_thresh: 不为background的IOU阈值
        :return: 指定head的损失，(5,), [coord_loss_xy, coord_loss_wh, noobj_iou_loss, obj_iou_loss, class_loss]
        """
        # 0. 获取3个head中和gt最大的IOU和对应的坐标 [[H, W, B], ...]
        response_max_iou = tf.gather_nd(response_max_iou, max_pos)
        target_grid_xyz = tf.gather_nd(target_grid_xyz, max_pos)
    
        # 1. 得到 object_mask 和 background_mask
        # 根据response anchor的位置，得到负责前景目标的grid、anchor的mask，(H, W, B)
        object_mask = tf.sparse_to_dense(sparse_indices=target_grid_xyz, output_shape=[height, width, box_num],
                                         sparse_values=1.0, default_value=0.0, validate_indices=False)
        # 小于IOU阈值且不为最大IOU的预测anchor，归类为background，统计mask，(H, W, B)
        background_mask = tf.cast(max_iou < iou_thresh, dtype=tf.float32)
        background_mask = tf.multiply(background_mask, (1 - object_mask))
        # 2. 计算各项损失：
        # 2.1 计算背景IOU损失：loss(noobj_iou)
        # check一下learning note里的CE与MSE的比较，注意CE的loss比MSE要大，原论文用MSE该项损失权重是0.5
        # 我们如果用CE，那么损失权重是不是应该更小？才能平衡该项loss与其他loss的影响
        # noobj_iou_loss = tf.square(predict[:, :, :, 4])  # L2损失函数
        noobj_iou_loss = - tf.log(1 - predict[:, :, :, 4])  # CE损失函数
        if self.is_focal_loss:
            noobj_iou_loss = noobj_iou_loss * tf.pow(predict[:, :, :, 4], self.focal_gamma)
        noobj_iou_loss = self.noobj_weight[head_index] * tf.reduce_sum(noobj_iou_loss * background_mask)
        # 2.2 计算response anchor的IOU损失：loss(obj_iou)
        # 取由该head负责的target和对应的response预测
        response_target = tf.gather_nd(target, max_pos)
        response_pred = tf.gather_nd(predict, target_grid_xyz)
        # obj_iou_loss(score)，可以以IOU=1为ground truth，也可使用真实IOU(response_max_iou)为gt
        # obj_iou_loss = tf.square(1 - response_pred[:, 4])  # L2损失函数
        obj_iou_loss = - tf.log(response_pred[:, 4])  # CE损失函数
        if self.is_focal_loss:
            obj_iou_loss = obj_iou_loss * (tf.pow(1 - response_pred[:, 4], self.focal_gamma) * self.focal_alpha)
        obj_iou_loss = self.obj_weight[head_index] * tf.reduce_sum(obj_iou_loss)  # CE损失函数
        # 2.3 计算坐标损失：loss(xy) + loss(wh)
        coord_loss_xy = self.coord_xy_weight[head_index] * tf.reduce_sum(tf.square(response_target[:, 0:2] -
                                                                                   response_pred[:, 0:2]))
        coord_loss_wh = self.coord_wh_weight[head_index] * tf.reduce_sum(tf.square(tf.sqrt(response_target[:, 2:4]) -
                                                                                   tf.sqrt(response_pred[:, 2:4])))
        # 2.4 计算分类损失：loss(class)
        if self.class_num >= 1:
            targets_class_prob = tf.one_hot(tf.cast(response_target[:, 4], dtype=tf.int32), depth=self.class_num)
            # class_loss = tf.square(targets_class_prob - response_pred[:, 5:])  # L2损失
            class_loss = - targets_class_prob * tf.log(response_pred[:, 5:])  # CE损失函数
            class_loss = self.cls_weight[head_index] * tf.reduce_sum(class_loss, name='class_loss')
        else:
            class_loss = tf.constant(0.0, dtype=tf.float32)
    
        loss = tf.stack([coord_loss_xy, coord_loss_wh, noobj_iou_loss, obj_iou_loss, class_loss], axis=-1)
        return loss
