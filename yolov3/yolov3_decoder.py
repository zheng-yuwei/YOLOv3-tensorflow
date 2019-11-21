# -*- coding: utf-8 -*-
"""
File yolov3_decoder.py
@author:ZhengYuwei
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras


class YOLOv3Decoder(object):
    """ YOLO v3 解码 """
    
    def __init__(self, head_grid_sizes, class_num, anchor_boxes):
        """
        解码YOLO v3的输出
        :param head_grid_sizes: 待解码的YOLO v3的3个head的输出尺度
        :param class_num: 类别数
        :param anchor_boxes: 待解码的head的预定义anchors
        """
        # 每个head的grid尺寸
        head_8_grid_size, head_16_grid_size, head_32_grid_size = head_grid_sizes
        self.head_8_height, self.head_8_width = head_8_grid_size
        self.head_16_height, self.head_16_width = head_16_grid_size
        self.head_32_height, self.head_32_width = head_32_grid_size
        # YOLOv3结果每个head降采样后meshgrid的左上角坐标(left_top_x, left_top_y), (H, W, 1, 2)
        self.head_8_left_top = self._get_left_top(self.head_8_width, self.head_8_height)
        self.head_16_left_top = self._get_left_top(self.head_16_width, self.head_16_height)
        self.head_32_left_top = self._get_left_top(self.head_32_width, self.head_32_height)
        # anchor boxes
        head_8_anchor_boxes, head_16_anchor_boxes, head_32_anchor_boxes = anchor_boxes
        self.head_8_box_num = len(head_8_anchor_boxes)
        self.head_16_box_num = len(head_16_anchor_boxes)
        self.head_32_box_num = len(head_32_anchor_boxes)
        head_8_anchor_boxes = tf.constant(head_8_anchor_boxes, dtype=tf.float32)
        head_16_anchor_boxes = tf.constant(head_16_anchor_boxes, dtype=tf.float32)
        head_32_anchor_boxes = tf.constant(head_32_anchor_boxes, dtype=tf.float32)
        self.head_8_anchor_boxes = head_8_anchor_boxes * np.array([self.head_8_width, self.head_8_height])
        self.head_16_anchor_boxes = head_16_anchor_boxes * np.array([self.head_16_width, self.head_16_height])
        self.head_32_anchor_boxes = head_32_anchor_boxes * np.array([self.head_32_width, self.head_32_height])
        # (center_x, center_y, w, h, confidence, class0...classN-1)
        self.coord_num = 4  # (center_x, center_y, w, h)
        self.conf_num = 1  # confidence
        self.class_num = class_num  # (class0, ..., classN-1)
        self.box_len = self.coord_num + self.conf_num + self.class_num

    @staticmethod
    def _get_left_top(width, height):
        """
        降采样为(height, width)的网格特征图后，meshgrid的左上角坐标(left_top_x, left_top_y), (H, W, 1, 2)
        :param width: 网格宽度
        :param height: 网格高度
        :return: 网格的左上角坐标矩阵，(H, W, 1, 2)，对应(H, W, B, coord)
        """
        left_top_x, left_top_y = tf.meshgrid(tf.range(0, width), tf.range(0, height))
        left_top_x = tf.expand_dims(left_top_x, axis=-1)
        left_top_y = tf.expand_dims(left_top_y, axis=-1)
        left_top = tf.concat([left_top_x, left_top_y], axis=-1)
        left_top = tf.cast(tf.reshape(left_top, shape=[height, width, 1, 2]), dtype=tf.float32)
        return left_top
    
    def decode(self, predicts):
        """
        对YOLO v3预测矩阵进行拆分、解码，得到3个head的预测中心点、宽高（没有归一化），score，类别概率向量
        :param predicts: YOLO v3网络输出的预测，3个head组成的矩阵
                        (N, H/32, W/32, head_32_channel*16 + head_32_channel*4 + head_32_channel)矩阵
        :return: 解码前的坐标预测值[t_x, t_y, t_w, t_h], 维度为(N, H, W, B, 4)
                解码后的预测值[center_x, center_y, w, h, score, class_prob], 维度为(N, H, W, B, 4 + 1 + class_num)
                以及解码后的bounding prediction [left_top_x, left_top_y, right_bottom_x, right_bottom_y]，(N, H, W, B, 4)
        """
        with tf.name_scope('decode_predict'):
            # 拆分出3个head的预测输出
            [_head_8_predicts, _head_16_predicts, _head_32_predicts] = self._unpack(predicts)
            # 分别解码
            head_8_predicts, head_8_predicts_boxes = self._decode_single_head(_head_8_predicts,
                                                                              self.head_8_left_top,
                                                                              self.head_8_anchor_boxes)
            head_16_predicts, head_16_predicts_boxes = self._decode_single_head(_head_16_predicts,
                                                                                self.head_16_left_top,
                                                                                self.head_16_anchor_boxes)
            head_32_predicts, head_32_predicts_boxes = self._decode_single_head(_head_32_predicts,
                                                                                self.head_32_left_top,
                                                                                self.head_32_anchor_boxes)
        predicts = [(_head_8_predicts[..., 0:4], head_8_predicts, head_8_predicts_boxes),
                    (_head_16_predicts[..., 0:4], head_16_predicts, head_16_predicts_boxes),
                    (_head_32_predicts[..., 0:4], head_32_predicts, head_32_predicts_boxes)]
        return predicts
    
    def _unpack(self, predicts):
        """
        将reshape后合并的YOLO v3输出拆分，并reshape回原输出形状
        :param predicts: YOLO v3网络输出的预测，3个head组成的矩阵
                        (N, H/32, W/32, head_32_channel*16 + head_32_channel*4 + head_32_channel)矩阵
        :return: 3个head的实际输出矩阵:
                 stride为8的head：(N, H/8, W/8, head_8_box_num, box_len)
                 stride为16的head：(N, H/16, W/16, head_16_box_num, box_len)
                 stride为32的head：(N, H/32, W/32, head_32_box_num, box_len)
        """
        start_channel = 0
        end_channel = self.head_8_box_num * self.box_len * 16
        head_8_predicts = tf.reshape(predicts[..., start_channel:end_channel],
                                     shape=[-1, self.head_8_height, self.head_8_width,
                                            self.head_8_box_num, self.box_len])
    
        start_channel = end_channel
        end_channel = end_channel + self.head_16_box_num * self.box_len * 4
        head_16_predicts = tf.reshape(predicts[..., start_channel:end_channel],
                                      shape=[-1, self.head_16_height, self.head_16_width,
                                             self.head_16_box_num, self.box_len])
    
        start_channel = end_channel
        end_channel = end_channel + self.head_32_box_num * self.box_len
        head_32_predicts = tf.reshape(predicts[..., start_channel:end_channel],
                                      shape=[-1, self.head_32_height, self.head_32_width,
                                             self.head_32_box_num, self.box_len])
    
        return [head_8_predicts, head_16_predicts, head_32_predicts]
    
    def _decode_single_head(self, single_head_predicts, head_left_top, head_anchor_boxes):
        """
        解析单一个head输出的预测矩阵
        :param single_head_predicts: 单一个head输出的预测矩阵, (N, head_H, head_W, head_B, box_len)
        :param head_left_top: head的grid的左上角坐标
        :param head_anchor_boxes: head的anchor boxes
        :return: 解码后的预测值[center_x, center_y, w, h, score, class_prob], 维度为(N, H, W, B, 4 + 1 + class_num)
                 以及解码后的bounding prediction [left_top_x, left_top_y, right_bottom_x, right_bottom_y]，(N, H, W, B, 4)
        """
        xy = self._decode_xy(single_head_predicts, head_left_top)
        wh = self._decode_wh(single_head_predicts, head_anchor_boxes)
        score = self._decode_score(single_head_predicts)

        if self.class_num >= 1:
            class_prob = self._decode_class_prob(single_head_predicts)
            single_head_predicts = tf.concat([xy, wh, score, class_prob], axis=-1)
        else:
            single_head_predicts = tf.concat([xy, wh, score], axis=-1)

        half_wh = wh / 2
        predicts_boxes = tf.concat([xy - half_wh, xy + half_wh], axis=-1)
        return single_head_predicts, predicts_boxes
    
    @staticmethod
    def _decode_xy(predicts, left_top):
        """
        预测矩阵中，目标预测中心点(center_x, center_y)坐标的解码
        [(left_top_x + shift_x), (left_top_y + shift_y)]
        shift_x = sigmoid(predicts_x)
        shift_y = sigmoid(predicts_y)
        :param predicts: 预测矩阵 (N, H, W, B, box_len)
        :param left_top: 左上角坐标
        :return: 解码后，预测的坐标，(N, H, W, B, 2)
        """
        xy = tf.sigmoid(predicts[..., 0:2])
        xy = tf.clip_by_value(xy, keras.backend.epsilon(), 1 - keras.backend.epsilon())
        xy = tf.add(xy, left_top)
        return xy
    
    @staticmethod
    def _decode_wh(predicts, anchor_boxes):
        """
        预测矩阵中，目标预测框的宽长(predict_w, predict_h)的解码
        [predict_w, predict_h]， predict_w = box_w * e^(output_w), predict_h = box_h * e^(output_h)
        :param predicts: 预测矩阵 (N, H, W, B, box_len)
        :param anchor_boxes: 锚框的尺寸,[W, H]
        :return: 解码后，预测的宽/长，(N, H, W, B, 2)
        """
        wh = tf.exp(predicts[..., 2:4])
        wh = tf.multiply(wh, anchor_boxes)
        return wh
    
    @staticmethod
    def _decode_score(predicts):
        """
        预测矩阵中，解码目标概率*目标预测的IOU：Pr(object) * IOU = sigmoid(t)
        :param predicts: 预测矩阵 (N, H, W, B, 1)
        :return:解码后，预测Pr(object) * IOU，(N, H, W, B, 1)
        """
        conf = tf.sigmoid(predicts[..., 4:5])
        conf = tf.clip_by_value(conf, keras.backend.epsilon(), 1 - keras.backend.epsilon())
        return conf
    
    @staticmethod
    def _decode_class_prob(predicts):
        """
        预测矩阵中，目标预测类别概率向量(class_0, .., class_N-1)的解码
        :param predicts: 预测矩阵 (N, H, W, B, box_len)
        :return:解码后，预测类别概率向量，(N, H, W, B, class_num)
        """
        max_class_prob = tf.reduce_max(predicts[..., 5:], axis=-1, keepdims=True)
        class_prob = tf.nn.softmax(predicts[..., 5:] - max_class_prob, axis=-1)
        class_prob = tf.clip_by_value(class_prob, keras.backend.epsilon(), 1 - keras.backend.epsilon())
        return class_prob
