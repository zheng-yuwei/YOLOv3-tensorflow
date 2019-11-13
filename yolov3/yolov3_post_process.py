# -*- coding: utf-8 -*-
"""
File yolov3_post_process.py
@author:ZhengYuwei
"""
import numpy as np
import cv2


class YOLOv3PostProcessor(object):
    """ YOLO v3 后处理函数 """
    
    @staticmethod
    def filter_boxes(prediction, predict_boxes, score_thresh):
        """
        预测bounding boxes的阈值过滤，得到归一化尺度的numpy.array(k, 8)，[(left top right bottom iou prob class score)..]
        :param prediction: 待阈值过滤的预测输出，(W, H, B, 4 + 1 + class_num)的numpy数组
        :param predict_boxes: 从prediction提取出来得到的bounding boxes坐标，(W, H, B, 4)的numpy数组
        :param score_thresh: score的阈值
        :return 经过阈值过滤的归一化尺度的预测框，numpy.array(k, 8)，[x0, y0, x1, y1, iou, 概率, 预测类别, 得分]
        """
        height, width, box_num, box_len = prediction.shape
        all_score = prediction[:, :, :, 4]
        all_class_prob = np.ones_like(all_score)  # 默认类别概率为1
        all_class_indices = np.zeros_like(all_score)  # 默认类别为0
        if box_len > 5:
            all_class_prob = np.max(prediction[:, :, :, 5:], axis=-1)
            all_class_indices = np.argmax(prediction[:, :, :, 5:], axis=-1)
            all_score = all_class_prob * all_score
        
        # 保留大于阈值的、归一化的候选框，进行NMS
        high_score_position = np.where(np.reshape(all_score > score_thresh, [-1]))
        left_top_x = np.take(predict_boxes[:, :, :, 0], high_score_position) / width
        left_top_y = np.take(predict_boxes[:, :, :, 1], high_score_position) / height
        right_bottom_x = np.take(predict_boxes[:, :, :, 2], high_score_position) / width
        right_bottom_y = np.take(predict_boxes[:, :, :, 3], high_score_position) / height
        confidence = np.take(prediction[:, :, :, 4], high_score_position)
        class_prob = np.take(all_class_prob, high_score_position)
        class_indices = np.take(all_class_indices, high_score_position)
        high_score = np.take(all_score, high_score_position)
        # (k, 8), [x0, y0, x1, y1, iou, 概率, 预测类别， 得分=iou*概率]
        boxes = np.stack([left_top_x, left_top_y, right_bottom_x, right_bottom_y,
                          confidence, class_prob, class_indices, high_score], axis=-1)[0]
        return boxes
    
    @staticmethod
    def apply_nms(boxes, nms_thresh):
        """ 实施nms筛选候选框
        :param boxes: 归一化尺度的候选框列表，元素为numpy.array(8)，(x0, y0, x1, y1, iou, 概率, 预测类别, 得分)
        :param nms_thresh: NMS阈值
        :return 筛选后的boxes
        """
        sorted_boxes = sorted(boxes, key=lambda d: d[7], reverse=True)
        index, box_num = 0, len(sorted_boxes) - 1
        while index < box_num:
            same_class_boxes = [(index + 1 + i, box) for (i, box) in enumerate(sorted_boxes[(index + 1):])
                                if box[6] == sorted_boxes[index][6]]
            boxes_iou = [(i, YOLOv3PostProcessor._cal_iou(sorted_boxes[index], box)) for (i, box) in same_class_boxes]
            remove_item_counter = 0
            for i, iou in boxes_iou:
                if iou > nms_thresh:
                    del sorted_boxes[i - remove_item_counter]
                    remove_item_counter += 1
                    box_num -= 1
            index += 1
        return sorted_boxes
    
    @staticmethod
    def _cal_iou(box, truth):
        """ 两个box的iou
        :param box: 第一个box坐标，[left top right bottom]
        :param truth: 第二个box坐标，[left top right bottom]
        :return 两个box的iou
        """
        w = YOLOv3PostProcessor._overlap(box[0], box[2], truth[0], truth[2])
        h = YOLOv3PostProcessor._overlap(box[1], box[3], truth[1], truth[3])
        if w <= 0 or h <= 0:
            return 0
        inter_area = w * h
        union_area = (box[2] - box[0]) * (box[3] - box[1]) + (truth[2] - truth[0]) * (truth[3] - truth[1]) - inter_area
        return inter_area / union_area
    
    @staticmethod
    def _overlap(x1, x2, x3, x4):
        """ 两个box在指定维度上的重合长度
        :param x1: 第一个box指定维度上的起点
        :param x2: 第一个box指定维度上的终点
        :param x3: 第二个box指定维度上的起点
        :param x4: 第二个box指定维度上的终点
        :return 指定维度上的重合长度
        """
        left = max(x1, x3)
        right = min(x2, x4)
        return right - left
    
    @staticmethod
    def resize_boxes(boxes, target_size):
        """
        将bounding boxes按尺寸伸缩
        :param boxes: 归一化尺度的候选框列表，元素为numpy.array(8)，(x0, y0, x1, y1, iou, 概率, 预测类别, 得分)
        :param target_size: 目标尺寸
        :return: 目标尺度的候选框列表，元素为numpy.array(8)，(x0, y0, x1, y1, iou, 概率, 预测类别, 得分)
        """
        boxes = [np.concatenate([box[:4] * target_size, box[4:]], axis=-1) for box in boxes]
        return boxes
    
    @staticmethod
    def visualize(image, boxes, src_box_size, image_path):
        """
        :param image: 一帧类型为float32，范围为[0, 1]的图片（网络输入数据）
        :param boxes: src_box_size尺度下的预测框列表，numpy.array(k, 8)，[x0, y0, x1, y1, iou, 概率, 预测类别, 得分]
        :param src_box_size: 预测框的尺度
        :param image_path: 文件保存路径
        :return:
        """
        image = (255 * image).astype(np.uint8)
        image_height, image_width, _ = image.shape
        # 目标图形尺度，然后计算src_box_size尺度的bounding boxes伸缩到目标尺度的比例
        image_size = np.tile(np.array([image_width, image_height], dtype=np.float), [2])
        rescale_size = image_size / src_box_size
        for box in boxes:
            # 根据box计算得到原图的预测框
            left, top, right, bottom = box[:4] * rescale_size
            left = max(left, 0)
            top = max(top, 0)
            right = min(right, image_width)
            bottom = min(bottom, image_height)
            cv2.rectangle(image, (int(left), int(top)), (int(right), int(bottom)),
                          (255, 242, 35), max(1, int(3 * image_width / 1200)))
            cv2.putText(image, '{:d}|{:.2f}'.format(int(box[6]), box[7]),
                        (int(left), int(top)), cv2.FONT_HERSHEY_SIMPLEX,
                        max(0.3, 0.3 * image_width / 1000), (255, 0, 0))
        cv2.imwrite(image_path, image)
        return

