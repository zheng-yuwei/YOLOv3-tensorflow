# -*- coding: utf-8 -*-
"""
File bounding_box.py
@author:ZhengYuwei
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class BoundingBox(object):
    """ 定义BoundingBox类，描述bounding box的坐标及预定义一些操作 """
    
    def __init__(self, w, h):
        """ box参数初始化
        :param w: box的宽
        :param h: box的高
        """
        self.w = w
        self.h = h
        self.area = w * h
    
    @staticmethod
    def distance(box, boxes):
        """
        计算一个box与一列表boxes的 1 - IoUs 的静态方法 （作为距离）
        :param box: box对象
        :param boxes：box对象列表
        :return box对象与box对象列表中每一个元素的 1-IOU
        """
        boxes_w, boxes_h, boxes_area = map(np.array, zip(*[(box.w, box.h, box.area) for box in boxes]))
        intersections = np.minimum(box.w, boxes_w) * np.minimum(box.h, boxes_h)
        dist_iou = 1 - intersections / (box.area + boxes_area - intersections)
        return dist_iou
    
    @staticmethod
    def mean(boxes):
        """
        计算一个box对象列表的均值中心点
        :param boxes: box对象列表
        :return box对象列表的均值中心点box对象
        """
        boxes_w, boxes_h = map(np.array, zip(*[(box.w, box.h) for box in boxes]))
        return BoundingBox(boxes_w.mean(), boxes_h.mean())
    
    @staticmethod
    def plot(centroids, boxes, group, name='GT bounding prediction cluster'):
        """
        绘制boxes散点图：聚类中心点为黑色，其他每一类一种颜色
        :param centroids: 各个group中心点
        :param boxes: boxes列表
        :param group: boxes对应的归属的中心点
        :param name: 散点图名称
        :return:
        """
        col = ['black', 'silver', 'red', 'peru', 'yellow', 'green', 'cyan', 'blue', 'fuchsia', 'pink']
        # 中心点和boxes的 width 和 height 解析
        center_w, center_h = map(np.array, zip(*[(box.w, box.h) for box in centroids]))
        center_colors = col[:len(centroids)]  # 'k'
        boxes_w, boxes_h = map(np.array, zip(*[(box.w, box.h) for box in boxes]))
        boxes_colors = [col[i] for i in group]
        # 绘制散点图
        plt.grid(ls='--')
        plt.scatter(boxes_w, boxes_h, c=boxes_colors, s=36, alpha=0.3)
        plt.scatter(center_w, center_h, c=center_colors, marker='p', s=48, alpha=1)
        shift = np.max(center_h) * 0.1
        for box in centroids:
            plt.text(box.w, box.h - shift, '({:.3f}, {:.3f})'.format(box.w, box.h))
        plt.title(name)
        plt.xlabel('Width')
        plt.ylabel('Height')
        plt.show()
    
    @staticmethod
    def plot3D(centroids, boxes, group, cls_names, name='GT bounding prediction cluster'):
        """
        绘制boxes的3维散点图
        :param centroids: 各个group中心点
        :param boxes: boxes列表
        :param group: boxes对应的归属的中心点
        :param cls_names: 物体类别名称
        :param name: 散点图名称
        :return:
        """
        fig = plt.figure()
        ax = Axes3D(fig)
        col = ['black', 'silver', 'red', 'peru', 'yellow', 'green', 'cyan', 'blue', 'fuchsia', 'pink']
        # 中心点和boxes的 width 和 height 解析
        center_w, center_h = map(np.array, zip(*[(box.w, box.h) for box in centroids]))
        center_colors = 'k'  # col[:len(centroids)]
        boxes_w, boxes_h = map(np.array, zip(*[(box.w, box.h) for box in boxes]))
        boxes_colors = [col[i] for i in group]
        # 绘制散点图
        plt.grid(ls='--')
        ax.scatter(boxes_w, boxes_h, [0] * len(boxes_h), c=boxes_colors, alpha=0.3)
        ax.scatter(center_w, center_h, [0] * len(center_w), c=center_colors, marker='p', s=48, alpha=1)
        for index, cls_name in enumerate(cls_names):
            cls_indices = np.where([name == cls_name for name in cls_names])[0]
            ax.scatter(boxes_w[cls_indices], boxes_h[cls_indices], [(index + 1) * 10] * len(cls_indices),
                       c=[boxes_colors[i] for i in cls_indices], alpha=0.3)
            ax.scatter(center_w, center_h, [(index + 1) * 10] * len(center_w),
                       c=center_colors, marker='p', s=48, alpha=1)
        plt.title(name)
        ax.set_xlabel('Width')
        ax.set_ylabel('Height')
        ax.set_zlabel('Classes')
        ax.set_zlim(0, 10 * len(cls_names) + 10)
        plt.savefig("scatter3D.png")
        plt.show()
    
    @staticmethod
    def plot_pareto(centroids, boxes, group, title='IOU-Ratio Curve'):
        """
        绘制 IOU-probability曲线
        :param centroids: 聚类中心
        :param boxes: box列表
        :param group: boxes对应的归属的中心点
        :param title: 标题
        :return:
        """
        plt.grid(ls='--')
        plt.title('{} Pareto'.format(title))
        col = ['black', 'silver', 'red', 'peru', 'yellow', 'green', 'cyan', 'blue', 'fuchsia', 'pink']
        for i, box in enumerate(centroids):
            pos = np.where(group == i)[0]
            x = 1 - np.sort(BoundingBox.distance(box, [boxes[j] for j in pos]))  # 因为距离是 1 - IOU
            y = np.arange(len(x)) / len(x)
            plt.plot(x, y, color=col[i], label='cluster {}'.format(i))
        
        plt.legend()
        plt.xlabel('IoU')
        plt.ylabel('Sample ratio')
        plt.show()
