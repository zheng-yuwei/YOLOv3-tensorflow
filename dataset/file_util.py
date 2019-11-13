# -*- coding: utf-8 -*-
"""
File file_util.py
@author:ZhengYuwei
"""
import os
import logging
import functools
import tensorflow as tf

from dataset.dataset_util import DatasetUtil


class FileUtil(object):
    """
    从标签文件中，构造返回(image, label)的tf.data.Dataset数据集
    标签文件内容如下：
    image_name label0,label1,label2,...
    """
    
    @staticmethod
    def _parse_string_line(string_line, root_path):
        """
        解析文本中的一行字符串行，得到图片路径（拼接图片根目录）和标签
        :param string_line: 文本中的一行字符串，image_name label0 label1 label2 label3 ...
        :param root_path: 图片根目录
        :return: DatasetV1Adapter<(图片路径Tensor(shape=(), dtype=string)，标签Tensor(shape=(?,), dtype=float32))>
        """
        strings = tf.string_split([string_line], delimiter=' ').values
        image_path = tf.string_join([root_path, strings[0]], separator=os.sep)
        labels = tf.string_to_number(strings[1:], out_type=tf.float32)
        return image_path, labels
    
    @staticmethod
    def _parse_image_with_label(image_path, label, image_size):
        """
        根据图片路径和标签，读取图片
        :param image_path: 图片路径, Tensor(shape=(), dtype=string)
        :param label: 标签Tensor(shape(?,), dtype=float32))，本函数只产生图像dataset，故不需要
        :param image_size: 图像需要resize到的大小，(H, W)
        :return: 归一化的图片 Tensor(shape=(image_size[0], image_size[1], ?), dtype=float32)，与图片一起原比例缩放的标签文件
        """
        # 图片读取
        image = tf.read_file(image_path)
        image = tf.image.decode_jpeg(image)
        # 图片原比例resize，标签文件需要跟着变换
        src_size_hw = tf.cast(tf.shape(image)[0:2], dtype=tf.float32) / tf.cast(image_size, dtype=tf.float32)
        resize_wh_ratio = src_size_hw[::-1] / tf.reduce_max(src_size_hw)
        label = tf.reshape(label, shape=(-1, 5))
        label_xy = label[:, 0:2] * resize_wh_ratio + (1 - resize_wh_ratio) / 2.0
        label_wh = label[:, 2:4] * resize_wh_ratio
        label = tf.reshape(tf.concat([label_xy, label_wh, label[:, 4:]], axis=-1), shape=(-1,))
        
        image = tf.image.resize_image_with_pad(image, target_height=image_size[0], target_width=image_size[1],
                                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        
        # 这里使用tf.float32会将照片归一化，也就是 *1/255
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = tf.reverse(image, axis=[2])  # 读取的是rgb，需要转为bgr
        return image, label
    
    @staticmethod
    def get_dataset(file_path, root_path, image_size, batch_size, is_augment=True, is_test=False):
        """
        从标签文件读取数据，并解析为（image_path, labels)形式的列表
        标签文件内容格式为(图片路径 归一化的中心点x坐标 y坐标 归一化的宽w 高h 类别 ...)：
        image_name center_x center_y w h class ...（不定长的待检测目标标签，5的倍数长度）
        :param file_path: 标签文件路径
        :param root_path: 图片路径的根目录，用于和标签文件中的image_name拼接
        :param image_size: 图像需要resize到的尺寸
        :param batch_size: 批次大小
        :param is_augment: 是否对图片进行数据增强
        :param is_test: 是否为测试阶段，测试阶段的话，输出的dataset中多包含image_path
        :return: tf.data.Dataset对象
        """
        logging.info('利用标签文件、图片根目录生成tf.data数据集对象：')
        logging.info('1. 解析标签文件；')
        dataset = tf.data.TextLineDataset(file_path)
        dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=5 * batch_size))
        dataset = dataset.map(functools.partial(FileUtil._parse_string_line, root_path=root_path),
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        logging.info('2. 读取图片数据，构造image set和label set；')
        image_label_set = dataset.map(functools.partial(FileUtil._parse_image_with_label,
                                                        image_size=tf.constant(image_size, dtype=tf.int32)),
                                      num_parallel_calls=tf.data.experimental.AUTOTUNE)
        
        image_set = image_label_set.map(lambda image, _: image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        label_set = image_label_set.map(lambda _, label: label, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        
        if is_augment:
            logging.info('2.1 image set数据增强；')
            image_set = DatasetUtil.augment_image(image_set)
        
        # 构造batch
        image_set = image_set.batch(batch_size)
        # 由于不同的图片有不同数量的gt boxes数，所以标签长度不一致，需要pad一致：用-1进行pad
        label_set = label_set.padded_batch(batch_size, (tf.TensorShape([None])), padding_values=-1.)
        
        if is_test:
            logging.info('4. 完成tf.data (image, label, path) 测试数据集构造；')
            path_set = dataset.map(lambda image_path, label: image_path,
                                   num_parallel_calls=tf.data.experimental.AUTOTUNE)
            path_set = path_set.batch(batch_size)
            dataset = tf.data.Dataset.zip((image_set, label_set, path_set))
        else:
            logging.info('4. 完成tf.data (image, label) 训练数据集构造；')
            # 合并image、labels：
            # DatasetV1Adapter<shapes:((48,144,?), ((), ..., ())), types:(float32,(float32,...,flout32))>
            dataset = tf.data.Dataset.zip((image_set, label_set))
        
        logging.info('5. 构造tf.data多epoch训练模式；')
        # 缓存数据到内存
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset


def _test():
    import cv2
    import numpy as np
    import time
    
    # 开启eager模式进行图片读取、增强和展示
    tf.enable_eager_execution()
    np.random.seed(6)
    tf.set_random_seed(800)
    train_file_path = './test_sample/label.txt'  # 标签文件
    image_root_path = './test_sample/images'  # 图片根目录
    
    train_batch = 5
    image_size = (384, 480)
    train_set = FileUtil.get_dataset(train_file_path, image_root_path, image_size=image_size,
                                     batch_size=train_batch, is_augment=False)
    start = time.time()
    image_size = tf.tile(tf.constant(image_size[::-1], dtype=tf.float32), (2,))
    for count, dataset in enumerate(train_set):
        print('一批(%d)图像 shape：' % train_batch, dataset[0].shape)
        for image, labels in zip(*dataset):
            image, labels = np.array(image), np.array(labels)
            print(labels)
            labels = np.reshape(labels, newshape=(-1, 5))
            labels = labels[:, 0:4] * image_size
            
            labels = np.concatenate([labels[:, 0:2] - labels[:, 2:4] / 2, labels[:, 0:2] + labels[:, 2:4] / 2], axis=-1)
            for label in labels:
                cv2.rectangle(image, (int(label[0]), int(label[1])), (int(label[2]), int(label[3])), (255, 242, 35), 1)
            cv2.imshow('a', image)
            cv2.waitKey()
        
        if count == 100:
            break
    print('耗时：', time.time() - start)


if __name__ == '__main__':
    _test()
