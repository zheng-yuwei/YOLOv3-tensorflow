# -*- coding: utf-8 -*-
"""
File kmeans.py
@author:ZhengYuwei
"""
import numpy as np
from utils.anchors.bounding_box import BoundingBox


class KMeans(object):
    """ k-means算法 """
    
    def __init__(self, dataset, k, distance_func, mean_func):
        """
        k-means算法数据初始化
        :param dataset：待聚类数据
        :param k：中心点数量
        :param distance_func: 距离计算函数
        :param mean_func: 均值计算函数
        """
        self.dataset = dataset
        self.k = k
        self.num = len(dataset)
        self.distance_func = distance_func
        self.mean_func = mean_func
        self.centroids = [None] * k  # 聚类中心
        self.group = np.array([0] * self.num)  # 每个点归属的组的下标
        self.in_class_distance = 1e10  # 平均类内距离
    
    def cluster(self, iter_total, loss_delta_thresh=1e-6, method='k-means++'):
        """
        初始化聚类中心
        :param iter_total: 迭代计算聚类中心次数
        :param loss_delta_thresh: 停止迭代的损失（总最小距离）变化阈值
        :param method: 初始化方式
        :return 最后两次聚类迭代的平均类内距离误差
        """
        if method == 'k-means++':
            self.k_means_plus_init()
        elif method == 'k-means':
            # k-means方式初始化聚类中心（随机选取）
            centroid_indices = np.random.choice(self.num, self.k, replace=False)
            self.centroids = [self.dataset[i] for i in centroid_indices]
        else:
            raise ValueError('Wrong Parameter: method must be k-means++ or k-means!')
        
        iter_num = 0
        loss_delta = 1e10  # 初始化迭代次数和终止条件
        while loss_delta > loss_delta_thresh and iter_num < iter_total:
            iter_num += 1
            if iter_num % 100 == 0:
                print('Cluster: {}/{} done...'.format(iter_num, iter_total))
            # 根据聚类中心，分离数据集
            last_in_class_distance = self.in_class_distance
            self.in_class_distance = 0
            for index, point in enumerate(self.dataset):
                dists = self.distance_func(point, self.centroids)
                self.group[index] = np.argmin(dists)
                self.in_class_distance += dists[self.group[index]]
            self.in_class_distance /= self.num
            loss_delta = abs(last_in_class_distance - self.in_class_distance)
            # 重新计算聚类中心
            for index in range(self.k):
                self.centroids[index] = self.mean_func([self.dataset[int(i)] for i in np.where(self.group == index)[0]])
            # BoundingBox.plot(self.centroids, self.dataset, self.group, name='GT bounding prediction cluster')
        return loss_delta
    
    def k_means_plus_init(self):
        """ k-means++方式初始化聚类中心 """
        # 随机选取第一个聚类中心，后续依据离已选聚类中心的最短距离进行轮盘赌选择
        centroid_index = np.random.randint(self.num)
        self.centroids[0] = self.dataset[centroid_index]
        min_distance = self.distance_func(self.centroids[0], self.dataset)
        for index in range(1, self.k):
            # 若不是第一个聚类中心，需要求出离所有聚类中心最短距离，作为该点的评估距离
            if index != 1:
                min_distance = np.minimum(min_distance, self.distance_func(self.centroids[index - 1], self.dataset))
            # 轮盘赌选取下一个聚类中心：离所有聚类中心越远的点被选到的概率越大
            fitness = np.cumsum(min_distance)
            rnd_fitness = np.random.rand() * fitness[-1]
            for ind, val in enumerate(fitness):
                if val >= rnd_fitness:
                    self.centroids[index] = self.dataset[ind]
                    break
