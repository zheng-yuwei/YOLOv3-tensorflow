# -*- coding: utf-8 -*-
"""
File yolov3_post_process.py
@author:ZhengYuwei
"""
import numpy as np
import cv2


class YOLOv3PostProcessor(object):
    """
    YOLO v3 后处理函数
    1. 根据阈值，过滤各个head输出的预测bounding boxes；
    2. 跨head进行NMS，但是输出结果仍然维持3个head的bounding boxes；
    3. 将预测的bounding boxes resize到目标图像尺寸；
    4. 可视化：用不同颜色绘制每个head预测的结果：蓝，绿，红 对应 head /8, /16, /32。
    """
    HEAD_BOX_COLOR = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]  # 蓝，绿，红
    
    @staticmethod
    def filter_boxes(head_8_prediction, head_8_boxes, head_16_prediction, head_16_boxes,
                     head_32_prediction, head_32_boxes, score_thresh):
        """
        3个head预测boxes的阈值过滤，得到归一化尺度的numpy.array(k, 8)，[(left top right bottom iou prob class score)..]
        :param head_8_prediction: 待阈值过滤的head 8预测输出，(W, H, B, 4 + 1 + class_num)的numpy数组
        :param head_8_boxes: 从 head_8_prediction 提取出来得到的bounding boxes坐标，(W, H, B, 4)的numpy数组
        :param head_16_prediction: 待阈值过滤的head 16预测输出，(W, H, B, 4 + 1 + class_num)的numpy数组
        :param head_16_boxes: 从 head_16_prediction 提取出来得到的bounding boxes坐标，(W, H, B, 4)的numpy数组
        :param head_32_prediction: 待阈值过滤的head 32预测输出，(W, H, B, 4 + 1 + class_num)的numpy数组
        :param head_32_boxes: 从 head_32_prediction 提取出来得到的bounding boxes坐标，(W, H, B, 4)的numpy数组
        :param score_thresh: score的阈值
        :return: 经过阈值过滤的归一化尺度的预测框，numpy.array(k, 8)，[x0, y0, x1, y1, iou, 概率, 预测类别, 得分]
        """
        head_8_high_score_boxes = YOLOv3PostProcessor._filter_single_head_boxes(head_8_prediction, head_8_boxes,
                                                                                score_thresh)
        head_16_high_score_boxes = YOLOv3PostProcessor._filter_single_head_boxes(head_16_prediction, head_16_boxes,
                                                                                 score_thresh)
        head_32_high_score_boxes = YOLOv3PostProcessor._filter_single_head_boxes(head_32_prediction, head_32_boxes,
                                                                                 score_thresh)
        high_score_boxes = [head_8_high_score_boxes, head_16_high_score_boxes, head_32_high_score_boxes]
        return high_score_boxes
    
    @staticmethod
    def _filter_single_head_boxes(prediction, predict_boxes, score_thresh):
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
        if len(high_score_position[0]) == 0:
            return np.empty(shape=(0, 8), dtype=np.float)
        left_top_x = np.take(predict_boxes[:, :, :, 0], high_score_position) / width
        left_top_y = np.take(predict_boxes[:, :, :, 1], high_score_position) / height
        right_bottom_x = np.take(predict_boxes[:, :, :, 2], high_score_position) / width
        right_bottom_y = np.take(predict_boxes[:, :, :, 3], high_score_position) / height
        confidence = np.take(prediction[:, :, :, 4], high_score_position)
        class_prob = np.take(all_class_prob, high_score_position)
        class_indices = np.take(all_class_indices, high_score_position)
        high_score = np.take(all_score, high_score_position)
        # (k, 8), [x0, y0, x1, y1, iou, 概率, 预测类别， 得分=iou*概率]
        boxes = np.transpose(np.concatenate([left_top_x, left_top_y, right_bottom_x, right_bottom_y,
                                             confidence, class_prob, class_indices, high_score], axis=0))
        return boxes
    
    @staticmethod
    def apply_nms(boxes, nms_thresh):
        # 给三个head的bounding boxes加个标记位index，然后拼接
        start_index = 0
        for i, head_boxes in enumerate(boxes):
            end_index = len(head_boxes)
            if end_index == 0:
                boxes[i] = np.reshape(head_boxes, (0, 9))
                continue
            end_index += start_index
            indices = np.expand_dims(np.arange(start=start_index, stop=end_index, dtype=np.float), axis=-1)
            boxes[i] = np.concatenate([head_boxes, indices], axis=-1)
        # 应用nms
        high_score_boxes = np.concatenate(boxes, axis=0)
        sorted_boxes = YOLOv3PostProcessor._apply_nms(high_score_boxes, nms_thresh)
        # 保留nms后的box
        box_indices = set()
        for box in sorted_boxes:
            box_indices.add(box[-1])
        for i, head_boxes in enumerate(boxes):
            if len(head_boxes) == 0:
                continue
            new_head_boxes = list()
            for box in head_boxes:
                if box[-1] in box_indices:
                    new_head_boxes.append(box)
            boxes[i] = new_head_boxes
        return boxes
    
    @staticmethod
    def _apply_nms(boxes, nms_thresh):
        """
        实施nms筛选候选框
        :param boxes: 归一化尺度的候选框列表，元素为numpy.array(8)，(x0, y0, x1, y1, iou, 概率, 预测类别, 得分)
        :param nms_thresh: NMS阈值
        :return 筛选后的boxes
        """
        sorted_boxes = sorted(boxes, key=lambda d: d[7], reverse=True)
        index, box_num = 0, len(sorted_boxes) - 1
        while index < box_num:
            same_class_boxes = [(index + 1 + i, box) for (i, box) in enumerate(sorted_boxes[(index + 1):])
                                if box[6] == sorted_boxes[index][6]]
            boxes_iou = [(i, YOLOv3PostProcessor._cal_iou(sorted_boxes[index], box))
                         for (i, box) in same_class_boxes]
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
        """
        两个box的iou
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
        """
        两个box在指定维度上的重合长度
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
        boxes = [head_boxes if len(head_boxes) == 0 else
                 [np.concatenate([box[:4] * target_size, box[4:]], axis=-1) for box in head_boxes]
                 for head_boxes in boxes]
        return boxes
    
    @staticmethod
    def visualize(image, boxes, src_box_size, image_path):
        """
        可视化：将bounding boxes画到image中，然后保存到本地
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
        for i, head_boxes in enumerate(boxes):
            if len(head_boxes) == 0:
                continue
            for box in head_boxes:
                # 根据box计算得到原图的预测框
                left, top, right, bottom = box[:4] * rescale_size
                left = max(left, 0)
                top = max(top, 0)
                right = min(right, image_width)
                bottom = min(bottom, image_height)
                cv2.rectangle(image, (int(round(left)), int(round(top))), (int(round(right)), int(round(bottom))),
                              YOLOv3PostProcessor.HEAD_BOX_COLOR[i], max(1, round(3 * image_width / 1200)))
                cv2.putText(image, '{:.0f}|{:.2f}'.format(round(box[6]), box[7]),
                            (int(round(left)), int(round(top))), cv2.FONT_HERSHEY_SIMPLEX,
                            max(0.3, 0.3 * image_width / 1000), (255, 0, 0))
        cv2.imwrite(image_path, image)
        return

