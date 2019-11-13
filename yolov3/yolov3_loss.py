# -*- coding: utf-8 -*-
"""
File yolov3_loss.py
@author:ZhengYuwei
"""
import tensorflow as tf
from tensorflow import keras
from yolov3.yolov3_decoder import YOLOv3Decoder


class YOLOv3Loss(object):
    """ YOLOv3损失函数 """
    
    def __init__(self, head_grid_sizes, class_num, anchor_boxes, iou_thresh, loss_weights,
                 rectified_coord_num=0, rectified_loss_weight=0.01):
        """
        YOLO v3 损失函数参数的初始化
        :param head_grid_sizes: YOLO v3输出head的所有尺度列表
        :param class_num: 类别数
        :param anchor_boxes: YOLO v3的预定义anchors
        :param iou_thresh: 任一grid、anchor预测的bounding box与所有实际目标的IOU，小于该阈值且不为最大IOU则为background，
                            需要计算noobj情况下的IOU损失；大于该阈值且不为最大IOU则不计算损失
        :param loss_weights: 不同损失项的权重，[coord_loss_weight, noobj_loss_weight]
        :param rectified_coord_num: 前期给坐标做矫正损失的图片数
        :param rectified_loss_weight: 前期矫正坐标的损失的权重
        """
        self.height, self.width = head_grid_sizes
        self.output_wh = tf.constant([self.width, self.height], dtype=tf.float32)
        self.class_num = class_num
        self.iou_thresh = iou_thresh
        # 坐标损失项、背景IOU损失项权重
        self.coord_xy_weight, self.coord_wh_weight, self.noobj_weight, self.obj_weight, self.cls_weight = loss_weights
        self.coord_xy_weight = tf.constant(self.coord_xy_weight, dtype=tf.float32)
        self.coord_wh_weight = tf.constant(self.coord_wh_weight, dtype=tf.float32)
        self.noobj_weight = tf.constant(self.noobj_weight, dtype=tf.float32)
        self.obj_weight = tf.constant(self.obj_weight, dtype=tf.float32)
        self.cls_weight = tf.constant(self.cls_weight, dtype=tf.float32)
        # bounding box
        self.box_num = len(anchor_boxes)
        self.coord_num = 4
        self.conf_num = 1
        self.box_len = self.coord_num + self.conf_num + self.class_num
        # output decoder
        self.decoder = YOLOv3Decoder(head_grid_sizes, class_num, anchor_boxes)
        # 前期训练 rectified loss
        self.rectified_coord_num = rectified_coord_num  # 预定义的坐标校正图片数
        self.rectified_loss_weight = rectified_loss_weight
        # 前期矫正的图片数
        self.current_num = keras.backend.variable(0, dtype=tf.int32, name='current_image_num')
        # 用于记录损失函数的细节
        with tf.variable_scope('loss_detail'):
            self.rectified_coord_loss = tf.get_variable('rectified_coord_loss', shape=[],
                                                        initializer=tf.constant_initializer(0.0))
            self.coord_loss_xy = tf.get_variable('coord_loss_xy', shape=[], initializer=tf.constant_initializer(0.0))
            self.coord_loss_wh = tf.get_variable('coord_loss_wh', shape=[], initializer=tf.constant_initializer(0.0))
            self.noobj_iou_loss = tf.get_variable('noobj_iou_loss', shape=[], initializer=tf.constant_initializer(0.0))
            self.obj_iou_loss = tf.get_variable('obj_iou_loss', shape=[], initializer=tf.constant_initializer(0.0))
            self.class_loss = tf.get_variable('class_loss', shape=[], initializer=tf.constant_initializer(0.0))
    
    def loss(self, targets, predicts):
        """
        计算YOLO v3损失函数
        :param targets: 尺寸为(N, obj_num * 5)，每一行为：obj_num * [x, y, w, h, class],
                        (x, y): 归一化中心点坐标
                        (w, h): 归一化宽高
                        class: 目标所属类别标签
                        obj_num: 单张图中目标个数
        :param predicts: YOLO v3网络输出的预测，(N, H, W, B*(4+1+class_num))
        :return: 总损失
        """
        # 0. YOLO源码技巧：train-from-scratch 前期（12800 pic）坐标回归时，将预测wh回归为anchor，预测xy回归为grid的中心点：
        # w_{anchor} * exp(t_{wh})= w_{anchor} => t_{wh} = 0
        # left_top + sigmoid(t_{xy}) = left_top + 0.5 => t_{xy} = 0
        # 这样的好处是，有一个好的初始预测框，并且loss的梯度在最大处
        def get_rectified_coord_loss():
            nonlocal predicts
            with tf.control_dependencies([keras.backend.update_add(self.current_num, tf.shape(predicts)[0])]):
                predicts = tf.reshape(predicts, shape=[-1, self.height, self.width, self.box_num, self.box_len])
                rectified_loss = tf.reduce_sum(tf.square(predicts[:, :, :, :, 0:4]), axis=[1, 2, 3, 4])
                rectified_loss = tf.multiply(self.rectified_loss_weight, rectified_loss)
                rectified_loss = tf.reduce_mean(rectified_loss, keepdims=True)
            return rectified_loss
        # 1. 解码网络的输出
        # 预测值[x, y, w, h, score, class_prob], 维度为(N, H, W, B, 2 + 2 + 1 + class_num)
        # 以及解码后的bounding prediction [left_top_x, left_top_y, right_bottom_x, right_bottom_y]，(N, H, W, B, 4)
        decode_predicts, predicts_boxes = self.decoder.decode(predicts)
        # 2. 解码标签：(N, obj_num, 5)的标签矩阵, (N, obj_num, 4)的bounding boxes坐标
        targets, targets_boxes = self._decode_target(targets)
        # 3. 逐张图片计算损失函数，(N, 4)，按样本维度遍历
        yolov2_loss = tf.map_fn(lambda inp: self._single_image_loss(inp[0], inp[1], inp[2], inp[3]),
                                (targets, targets_boxes, decode_predicts, predicts_boxes), dtype=tf.float32,
                                parallel_iterations=1)
        yolov2_loss = tf.reduce_mean(yolov2_loss, axis=0)
        # 4. 汇总并记录所有损失 (6,)
        # 用于记录损失函数的细节，在logger_callback里使用
        update_op = [
            self.coord_loss_xy.assign(yolov2_loss[0]),
            self.coord_loss_wh.assign(yolov2_loss[1]),
            self.noobj_iou_loss.assign(yolov2_loss[2]),
            self.obj_iou_loss.assign(yolov2_loss[3]),
            self.class_loss.assign(yolov2_loss[4]),
        ]
        # 前期矫正的图片数小于预定义的坐标校正图片数，则继续加坐标校正损失
        # [rectified_coord_loss, coord_loss_xy, coord_loss_wh, noobj_iou_loss, obj_iou_loss, class_loss]
        total_loss = tf.cond(self.current_num <= self.rectified_coord_num,
                             lambda: tf.concat([yolov2_loss, get_rectified_coord_loss()], axis=-1),
                             lambda: yolov2_loss)
        update_op = tf.cond(self.current_num <= self.rectified_coord_num,
                            lambda: update_op + [self.rectified_coord_loss.assign(total_loss[5])],
                            lambda: update_op + [self.rectified_coord_loss.assign(0.0)])
        # 4. 汇总所有损失
        with tf.control_dependencies(update_op):
            total_loss = tf.reduce_sum(total_loss)
        return total_loss
    
    def _decode_target(self, targets):
        """
        解码标签数据
        :param targets: (N, (center_x, center_y, width, height, class) * obj_num)
                    其中，(center_x, center_y)是归一化的目标中心点，(width, height)是归一化的物体宽/高，class是物体类别
        :return: (N, obj_num, 5)的标签矩阵, (N, obj_num, 4)的bounding boxes坐标
        """
        # 目标label
        with tf.name_scope('decode_target'):
            targets = tf.reshape(targets, shape=[tf.shape(targets)[0], -1, 5])
            targets_xy = tf.multiply(targets[:, :, 0:2], self.output_wh, name='targets_xy')
            targets_wh = tf.multiply(targets[:, :, 2:4], self.output_wh, name='targets_wh')
            targets_prob = targets[:, :, 4:5]
            targets = tf.concat([targets_xy, targets_wh, targets_prob], axis=-1, name='concat_target_head')
            
            half_wh = targets_wh / 2
            targets_boxes = tf.concat([targets_xy - half_wh, targets_xy + half_wh],
                                      axis=-1, name='targets_bounding_boxes')
        return targets, targets_boxes
    
    def _single_image_loss(self, target, target_boxes, predict, predict_boxes):
        """
        单张图片的损失函数
        :param target: [[enter_x, center_y, width, height, class] ...]，(obj_num, 5)
        :param target_boxes: [[left_top_x, left_top_y, right_bottom_x, right_bottom_y] ...]，(obj_num, 4)
        :param predict: (H, W, B, 4 + 1 + class_num)
        :param predict_boxes: (H, W, B, 4), [[left_top_x, left_top_y, right_bottom_x, right_bottom_y]...]
        :return: 该图片的损失
        """
        # 1. 取出有效目标label：(obj_num, 5), (obj_num, 4) => (valid_num, 5), (valid_num, 4)
        target, target_boxes, valid_obj_num = self._get_valid_target(target, target_boxes)
        # 2. 计算IOU及response anchor的位置：
        # 所有grid、所有anchor预测框和所有实际目标的IOU，统计各anchor的最大IOU，(H, W, B)
        # 中心点grid所有anchor与对应的实际物体的最大IOU (valid_num,)，及其坐标[[H, W, B] * valid_num]
        with tf.name_scope('calculate_iou'):
            max_iou, response_max_iou, target_grid_xyz = self._calc_iou(target, target_boxes,
                                                                        predict, predict_boxes, valid_obj_num)
        # 3. 得到 object_mask 和 background_mask
        # 根据response anchor的位置，得到负责前景目标的grid、anchor的mask，(H, W, B)
        object_mask = tf.sparse_to_dense(sparse_indices=target_grid_xyz,
                                         output_shape=[self.height, self.width, self.box_num],
                                         sparse_values=1.0, default_value=0.0, validate_indices=False,
                                         name='object_mask')
        # 小于IOU阈值且不为最大IOU的预测anchor，归类为background，统计mask，(H, W, B)
        background_mask = tf.cast(max_iou < self.iou_thresh, dtype=tf.float32)
        background_mask = tf.multiply(background_mask, (1 - object_mask), name='background_mask')
        # 4. 计算各项损失：
        # 计算背景IOU损失：loss(noobj_iou)
        # check一下learning note里的CE与MSE的比较，注意CE的loss比MSE要大，原论文用MSE该项损失权重是0.5
        # 我们如果用CE，那么损失权重是不是应该更小？才能平衡该项loss与其他loss的影响
        # noobj_iou_loss = tf.square(predict[:, :, :, 4])  # L2损失函数
        noobj_iou_loss = - tf.log(1 - predict[:, :, :, 4])  # CE损失函数
        noobj_iou_loss = self.noobj_weight * tf.reduce_sum(noobj_iou_loss * background_mask)
        # 取每个待检测物体的response预测
        response_pred = tf.gather_nd(predict, target_grid_xyz)
        # obj_iou_loss(score)，可以以IOU=1为ground truth，也可使用真实IOU(response_max_iou)为gt
        # obj_iou_loss = tf.square(1 - response_pred[:, 4])  # L2损失函数
        obj_iou_loss = - tf.log(response_pred[:, 4])  # CE损失函数
        obj_iou_loss = self.obj_weight * tf.reduce_sum(obj_iou_loss, name='obj_iou_loss')  # CE损失函数
        # loss(xy) + loss(wh)
        coord_loss_xy = self.coord_xy_weight * tf.reduce_sum(tf.square(target[:, 0:2] - response_pred[:, 0:2]))
        coord_loss_wh = tf.reduce_sum(tf.square(tf.sqrt(target[:, 2:4]) - tf.sqrt(response_pred[:, 2:4])))
        coord_loss_wh = self.coord_wh_weight * coord_loss_wh
        # loss(class)
        if self.class_num >= 1:
            targets_class_prob = tf.one_hot(tf.cast(target[:, 4], dtype=tf.int32), depth=self.class_num)
            # class_loss = tf.square(targets_class_prob - response_pred[:, 5:])  # L2损失
            class_loss = - targets_class_prob * tf.log(response_pred[:, 5:])  # CE损失函数
            class_loss = self.cls_weight * tf.reduce_sum(class_loss, name='class_loss')
        else:
            class_loss = tf.constant(0.0, dtype=tf.float32)
            
        loss = tf.stack([coord_loss_xy, coord_loss_wh, noobj_iou_loss, obj_iou_loss, class_loss], axis=-1)
        return loss
    
    @staticmethod
    def _get_valid_target(target, target_boxes):
        """
        根据标签的类别列，判断是否为padding的数据，以取出有效目标label
        :param target: (obj_num, 5),[[center_x, center_y, width, height, class]...]
        :param target_boxes: (obj_num, 4), [[left_top_x, left_top_y, right_bottom_x, right_bottom_y] ...]
        :return: 保留有效目标label，分别将target和target_boxes的obj_num，降为valid_obj_num
        """
        # 我在tf.data加入padding -1使得每张图的标签一样长，得以构造tf.date.Dataset数据，可查看dataset/file_util.py
        valid_obj_mask = target[:, 0] >= 0
        valid_obj_num = tf.count_nonzero(valid_obj_mask)
        valid_obj_indices = tf.where(valid_obj_mask)
        target = tf.gather_nd(target, valid_obj_indices)
        target_boxes = tf.gather_nd(target_boxes, valid_obj_indices)
        return target, target_boxes, valid_obj_num
    
    @staticmethod
    def _calc_iou(target, target_boxes, predict, predict_boxes, valid_obj_num):
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
        # 1. 所有grid、所有anchor的预测框面积，取出真实物体中心点grid的所有anchor的预测框面积
        predict_area = predict[:, :, :, 2] * predict[:, :, :, 3]  # (H, W, B)
        
        response_grid_xy = tf.cast(tf.floor(target[:, 0:2]), dtype=tf.int32)
        response_grid_xy = tf.reverse(response_grid_xy, axis=[-1])  # 由坐标获取下标：[center_x, center_y]=>(H, W)
        response_area = tf.gather_nd(predict_area, response_grid_xy)  # 获得response grid的预测输出， (valid_num, B)
        # 2. 真实框面积
        target_area = target[:, 2] * target[:, 3]  # (valid_num,)
        # 3. 所有grid的所有anchor的预测框与真实框相交的面积，中心点grid的所有anchor预测框与对应gt框的相交面积
        tile_predict_boxes = tf.tile(tf.expand_dims(predict_boxes, axis=-2), [1, 1, 1, valid_obj_num, 1])
        left_top = tf.maximum(tile_predict_boxes[:, :, :, :, 0:2], target_boxes[:, 0:2])
        right_bottom = tf.minimum(tile_predict_boxes[:, :, :, :, 2:4], target_boxes[:, 2:4])
        inter_wh = right_bottom - left_top
        inter_area = inter_wh[:, :, :, :, 0] * inter_wh[:, :, :, :, 1]  # (H, W, B, valid_num)
        
        response_boxes = tf.gather_nd(predict_boxes, response_grid_xy)  # 获得response grid预测的boxes，(valid_num, B, 4)
        expand_target_boxes = tf.expand_dims(target_boxes, axis=1)
        response_left_top = tf.maximum(response_boxes[:, :, 0:2], expand_target_boxes[:, :, 0:2])
        response_right_bottom = tf.minimum(response_boxes[:, :, 2:4], expand_target_boxes[:, :, 2:4])
        response_inter_wh = response_right_bottom - response_left_top
        response_inter_area = response_inter_wh[:, :, 0] * response_inter_wh[:, :, 1]  # (valid_num, B)
        # 4. 计算IOU
        # 所有grid的所有anchor与所有gt的IOU，gt中最大的IOU作为该anchor的IOU
        predict_area = tf.tile(tf.expand_dims(predict_area, axis=-1), [1, 1, 1, valid_obj_num])
        iou = inter_area / (predict_area + target_area - inter_area)
        max_iou = tf.reduce_max(iou, axis=-1, keepdims=False)  # (H, W, B)
        # 中心点grid所有anchor与对应的实际物体的最大IOU (valid_num,)，及其坐标[[H, W, B] * valid_num]
        response_iou = response_inter_area / (response_area + tf.expand_dims(target_area, axis=-1) - response_inter_area)
        response_max_iou = tf.reduce_max(response_iou, axis=-1, keepdims=False)
        max_arg = tf.arg_max(response_iou, dimension=-1, output_type=tf.int32)
        response_grid_xyz = tf.concat([response_grid_xy, tf.expand_dims(max_arg, axis=-1)], axis=-1)
        return max_iou, response_max_iou, response_grid_xyz
