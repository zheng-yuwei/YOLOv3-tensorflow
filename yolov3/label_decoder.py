# -*- coding: utf-8 -*-
"""
Created on 2019/11/17
File label_decoder
@author:ZhengYuwei
"""
import tensorflow as tf


class LabelDecoder(object):
    """ YOLO v3标签解码器 """
    
    def __init__(self, head_grid_sizes):
        """
        解码YOLO v3的标签
        :param head_grid_sizes: 待解码的YOLO v3的3个head的输出尺度
        """
        head_8_grid_size, head_16_grid_size, head_32_grid_size = head_grid_sizes
        self.head_8_height, self.head_8_width = head_8_grid_size
        self.head_16_height, self.head_16_width = head_16_grid_size
        self.head_32_height, self.head_32_width = head_32_grid_size
        self.head_8_wh = tf.constant([self.head_8_width, self.head_8_height], dtype=tf.float32)
        self.head_16_wh = tf.constant([self.head_16_width, self.head_16_height], dtype=tf.float32)
        self.head_32_wh = tf.constant([self.head_32_width, self.head_32_height], dtype=tf.float32)
        
    def decode(self, targets):
        """
        解码标签数据
        :param targets: (N, (center_x, center_y, width, height, class) * obj_num)
                    其中，(center_x, center_y)是归一化的目标中心点，(width, height)是归一化的物体宽/高，class是物体类别
        :return: (N, obj_num, 5)的标签矩阵, (N, obj_num, 4)的bounding boxes坐标
        """
        # 解析目标label
        with tf.name_scope('decode_target'):
            targets = tf.reshape(targets, shape=[tf.shape(targets)[0], -1, 5])
            head_8_targets, head_8_targets_boxes = self._decode_single_head(targets, self.head_8_wh)
            head_16_targets, head_16_targets_boxes = self._decode_single_head(targets, self.head_16_wh)
            head_32_targets, head_32_targets_boxes = self._decode_single_head(targets, self.head_32_wh)
        decode_targets = [(head_8_targets, head_8_targets_boxes),
                          (head_16_targets, head_16_targets_boxes),
                          (head_32_targets, head_32_targets_boxes)]
        return decode_targets
    
    @staticmethod
    def _decode_single_head(targets, head_wh):
        """
        将label解码为单个head的label
        :param targets: (N, (center_x, center_y, width, height, class) * obj_num)
                    其中，(center_x, center_y)是归一化的目标中心点，(width, height)是归一化的物体宽/高，class是物体类别
        :param head_wh: 指定head的宽高
        :return: 指定head的(N, obj_num, 5)标签矩阵, (N, obj_num, 4) bounding boxes坐标
        """
        targets_xy = tf.multiply(targets[:, :, 0:2], head_wh)
        targets_wh = tf.multiply(targets[:, :, 2:4], head_wh)
        targets_prob = targets[:, :, 4:5]
        targets = tf.concat([targets_xy, targets_wh, targets_prob], axis=-1)
    
        half_wh = targets_wh / 2
        targets_boxes = tf.concat([targets_xy - half_wh, targets_xy + half_wh], axis=-1)
        return targets, targets_boxes
