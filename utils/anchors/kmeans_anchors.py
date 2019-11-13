# -*- coding: utf-8 -*-
"""
File kmeans_anchors.py
@author:ZhengYuwei
"""
import os
import numpy as np
from utils.anchors.bounding_box import BoundingBox
from utils.anchors.kmeans import KMeans


def parse(label_file_path):
    """
    解析标签文件
    :param label_file_path: image_path x0 y0 w0 h0 cls ... 的标签文件
    :return: 得到bounding box宽高组成的数组(-1, 2)，类别数组
    """
    if not os.path.isfile(label_file_path):
        raise ValueError('请输入标签文件路径！')
    # 解析得到宽高、类别
    err_line_no = []  # 异常数据行
    xywh_cls = []
    with open(label_file_path, 'r') as file:
        for line_no, line in enumerate(file):
            bboxes = line.strip().split(' ')[1:]
            if len(bboxes) % 5 != 0:
                err_line_no.append(line_no)
                continue
            bboxes = np.array([float(num) for num in bboxes])
            bboxes = np.reshape(bboxes, [-1, 5])
            for box in bboxes:
                xywh_cls.append(box)
    xywh_cls = np.array(xywh_cls)
    object_wh = xywh_cls[:, 2:4]
    object_class = xywh_cls[:, 4].astype(np.int)
    # 将宽高构造为BoundingBox对象
    boxes_list = [None] * object_wh.shape[0]
    for index, wh in enumerate(object_wh):
        boxes_list[index] = BoundingBox(wh[0], wh[1])
    return boxes_list, object_class


def total_anchors(object_boxes, object_class_name=None, class_name_set=None):
    """
    对所有待检测目标进行聚类，求出聚类中心，并绘制散点图
    :param object_boxes: 所有物体的boxes
    :param object_class_name: 对应的boxes的物体类别的名称
    :param class_name_set: 物体类别名称的集合
    :return 聚类中心
    """
    # 开始聚类
    cluster_machine = KMeans(object_boxes, k=5, distance_func=BoundingBox.distance, mean_func=BoundingBox.mean)
    res = cluster_machine.cluster(iter_total=500)
    print('最后的聚类改进步长效果为:{:.8f}'.format(res))
    # 绘制聚类结果
    '''
    Box.plot3D(cluster_machine.centroids, object_boxes, cluster_machine.group, object_class_name,
               name='GT bounding prediction cluster({})'.format(len(object_boxes)))
    '''
    # 绘制聚类后所有点的散点图
    BoundingBox.plot(cluster_machine.centroids, object_boxes, cluster_machine.group,
                     name='GT bounding prediction cluster({})'.format(len(object_boxes)))
    BoundingBox.plot_pareto(cluster_machine.centroids, object_boxes, cluster_machine.group)
    # 绘制各个子类聚类后的散点图
    if object_class_name is not None:
        for class_name in class_name_set:
            pos = np.where([name == class_name for name in object_class_name])[0]
            BoundingBox.plot(cluster_machine.centroids, [object_boxes[i] for i in pos], cluster_machine.group[pos],
                             name='GT bounding prediction cluster({}/{})'.format(class_name, len(pos)))
    
    # 聚类中心
    centers = [(box.w, box.h) for box in cluster_machine.centroids]
    print('聚类中心[w, h]:', centers)
    print('类别分组:', cluster_machine.group)
    print('与中心点的平均距离误差:', cluster_machine.in_class_distance)
    return centers


if __name__ == '__main__':
    # 1. 解析标签文件，构造为BoundingBox对象
    label_file = '../../dataset/test_sample/label.txt'
    bounding_boxes, classes = parse(label_file)
    print('bounding boxes数量: ', len(bounding_boxes))
    
    # 2. 类别名称，为画图做准备（这部分可不需要）
    class_names, cls_name_set = None, None
    # cls_name_set = ['dog', 'person', 'train', 'sofa', 'chair', 'car', 'horse', 'cat', 'cow', 'bus',
    #                 'bicycle', 'pottedplant', 'diningtable']
    # class_names = [None] * len(classes)
    # for index, num in enumerate(classes):
    #     class_names[index] = cls_name_set[num]
    # cls_name_set = set(cls_name_set)
    # print('检测物体类名: ', cls_name_set)
    
    # 3. 对所有boxes进行聚类
    centers = total_anchors(bounding_boxes, class_names, cls_name_set)
