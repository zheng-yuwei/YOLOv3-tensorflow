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
    
    def __init__(self, grid_size, class_num, anchor_boxes):
        """
        解码YOLO v2的输出
        :param grid_size: YOLO v3输出尺度
        :param class_num: 类别数
        :param anchor_boxes: YOLO v3的预定义anchors
        """
        self.height, self.width = grid_size
        # B * (center_x, center_y, w, h, confidence, class0...classN-1)
        self.box_num = len(anchor_boxes)
        self.coord_num = 4
        self.conf_num = 1
        self.class_num = class_num
        self.box_len = self.coord_num + self.conf_num + self.class_num
        self.anchor_boxes = tf.constant(anchor_boxes, dtype=tf.float32) * np.array([self.width, self.height])
        # YOLOv2结果降采样后meshgrid的左上角坐标(left_top_x, left_top_y), (W, H)
        left_top_x, left_top_y = tf.meshgrid(tf.range(0, self.width), tf.range(0, self.height))
        left_top_x = tf.expand_dims(left_top_x, axis=-1)
        left_top_y = tf.expand_dims(left_top_y, axis=-1)
        left_top = tf.concat([left_top_x, left_top_y], axis=-1)
        self.left_top = tf.cast(tf.reshape(left_top, shape=[self.height, self.width, 1, 2]), dtype=tf.float32)
    
    def decode(self, predicts):
        """
        对YOLO v2预测矩阵进行解码，得到预测中心点、宽高（没有归一化），score，类别概率向量
        :param predicts: 预测矩阵 (N, H, W, B*box_len)
        :return: 解码后的预测值[x, y, w, h, score, class_prob], 维度为(N, H, W, B, 2 + 2 + 1 + class_num)
                  以及解码后的bounding prediction [left_top_x, left_top_y, right_bottom_x, right_bottom_y]，(N, H, W, B, 4)
        """
        with tf.name_scope('decode_predict'):
            predicts = tf.reshape(predicts, shape=[-1, self.height, self.width, self.box_num, self.box_len])
            xy = self._decode_xy(predicts)
            wh = self._decode_wh(predicts)
            score = self._decode_score(predicts)

            if self.class_num >= 1:
                class_prob = self._decode_class_prob(predicts)
                predicts = tf.concat([xy, wh, score, class_prob], axis=-1, name='concat_predict_head')
            else:
                predicts = tf.concat([xy, wh, score], axis=-1, name='concat_predict_head')
            
            half_wh = wh / 2
            predicts_boxes = tf.concat([xy - half_wh, xy + half_wh], axis=-1, name='predict_bounding_boxes')
        return predicts, predicts_boxes
    
    def _decode_xy(self, predicts):
        """
        预测矩阵中，目标预测中心点(predict_x, predict_y)坐标的解码
        [(left_top_x + shift_x), (left_top_y + shift_y)]
        shift_x = sigmoid(output_x)
        shift_y = sigmoid(output_y)
        :param predicts: 预测矩阵 (N, H, W, B, box_len)
        :return: 解码后，预测的坐标，(N, H, W, B, 2)
        """
        xy = tf.sigmoid(predicts[:, :, :, :, 0:2])
        xy = tf.clip_by_value(xy, keras.backend.epsilon(), 1 - keras.backend.epsilon())
        xy = tf.add(xy, self.left_top, name='predict_coord_xy')
        return xy
    
    def _decode_wh(self, predicts):
        """
        预测矩阵中，目标预测框的宽长(predict_w, predict_h)的解码
        [predict_w, predict_h]， predict_w = box_w * e^(output_w), predict_h = box_h * e^(output_h)
        :param predicts: 预测矩阵 (N, H, W, B, box_len)
        :return: 解码后，预测的宽/长，(N, H, W, B, 2)
        """
        wh = tf.exp(predicts[:, :, :, :, 2:4])
        wh = tf.multiply(wh, self.anchor_boxes, name='predict_coord_wh')
        return wh
    
    @staticmethod
    def _decode_score(predicts):
        """
        预测矩阵中，解码目标概率*目标预测的IOU：Pr(object) * IOU = sigmoid(t)
        :param predicts: 预测矩阵 (N, H, W, B, 1)
        :return:解码后，预测Pr(object) * IOU，(N, H, W, B, 1)
        """
        conf = tf.sigmoid(predicts[:, :, :, :, 4:5], name='predict_score')
        conf = tf.clip_by_value(conf, keras.backend.epsilon(), 1 - keras.backend.epsilon())
        return conf
    
    @staticmethod
    def _decode_class_prob(predicts):
        """
        预测矩阵中，目标预测类别概率向量(class_0, .., class_N-1)的解码
        :param predicts: 预测矩阵 (N, H, W, B, box_len)
        :return:解码后，预测类别概率向量，(N, H, W, B, class_num)
        """
        max_class_prob = tf.reduce_max(predicts[:, :, :, :, 5:], axis=-1, keepdims=True)
        class_prob = tf.nn.softmax(predicts[:, :, :, :, 5:] - max_class_prob, axis=-1, name='predict_class_prob')
        class_prob = tf.clip_by_value(class_prob, keras.backend.epsilon(), 1 - keras.backend.epsilon())
        return class_prob
