# -*- coding: utf-8 -*-
"""
File run.py
@author:ZhengYuwei
"""
import os
import logging
import numpy as np
import tensorflow as tf
from tensorflow import keras

from configs import FLAGS
from utils.logger import generate_logger
from dataset.file_util import FileUtil
from yolov3.trainer import YOLOv3Trainer
from yolov3.yolov3_decoder import YOLOv3Decoder
from yolov3.yolov3_post_process import YOLOv3PostProcessor

if FLAGS.mode in ('test', 'predict'):
    tf.enable_eager_execution()
if FLAGS.mode == 'train':
    keras.backend.set_learning_phase(True)
else:
    keras.backend.set_learning_phase(False)

keras.backend.set_epsilon(1e-8)
np.random.seed(6)
tf.set_random_seed(800)


def train(yolov3_trainer):
    """ YOLO v3模型训练 """
    logging.info('加载训练数据集：%s', FLAGS.train_label_path)
    train_dataset = FileUtil.get_dataset(FLAGS.train_label_path, FLAGS.train_set_dir,
                                         image_size=FLAGS.input_image_size[0:2],
                                         batch_size=FLAGS.batch_size, is_augment=FLAGS.is_augment, is_test=False)
    yolov3_trainer.train(train_dataset, None)
    logging.info('训练完毕！')


def test(yolov3_trainer, yolov3_decoder, save_path=None):
    """
    YOLO v3模型测试
    :param yolov3_trainer: yolov3检测模型
    :param yolov3_decoder: yolov3模型输出解码器
    :param save_path：测试结果图形报错路径
    """
    logging.info('加载测试数据集：%s', FLAGS.test_label_path)
    test_set = FileUtil.get_dataset(FLAGS.test_label_path, FLAGS.test_set_dir,
                                    image_size=FLAGS.input_image_size[0:2],
                                    batch_size=FLAGS.batch_size, is_augment=False, is_test=True)
    total_test = int(np.ceil(FLAGS.val_set_size / FLAGS.batch_size))
    input_box_size = np.tile(FLAGS.input_image_size[1::-1], [2])  # 网络输入尺度，[W, H, W, H]
    # images为转为[0,1]范围的float32类型的TensorFlow矩阵
    for batch_counter, (images, labels, image_paths) in enumerate(test_set):
        if batch_counter > total_test:
            break
        images, labels, image_paths = np.array(images), np.array(labels), np.array(image_paths)
        predictions = yolov3_trainer.predict(images)
        [(_, head_8_predicts, head_8_predicts_boxes),
         (_, head_16_predicts, head_16_predicts_boxes),
         (_, head_32_predicts, head_32_predicts_boxes)] = yolov3_decoder.decode(predictions)
        for image, label, image_path, head_8_prediction, head_8_boxes, \
            head_16_prediction, head_16_boxes, head_32_prediction, head_32_boxes in \
                zip(images, labels, image_paths, np.array(head_8_predicts), np.array(head_8_predicts_boxes),
                    np.array(head_16_predicts), np.array(head_16_predicts_boxes),
                    np.array(head_32_predicts), np.array(head_32_predicts_boxes)):
            # (k, 8)， 归一化尺度->网络输入尺度的[(left top right bottom iou prob class score) ... ]
            high_score_boxes = YOLOv3PostProcessor.filter_boxes(head_8_prediction, head_8_boxes,
                                                                head_16_prediction, head_16_boxes,
                                                                head_32_prediction, head_32_boxes,
                                                                FLAGS.confidence_thresh)
            nms_boxes = YOLOv3PostProcessor.apply_nms(high_score_boxes, FLAGS.nms_thresh)
            in_boxes = YOLOv3PostProcessor.resize_boxes(nms_boxes, target_size=input_box_size)
            if save_path is not None:
                image_path = os.path.join(save_path, str(os.path.basename(image_path), 'utf-8'))
                YOLOv3PostProcessor.visualize(image, in_boxes, src_box_size=input_box_size, image_path=image_path)
            # TODO 根据预测结果，计算AP，mAP
    return


def predict(yolov3_trainer, yolov3_decoder, image_paths, save_path):
    """
    YOLO v3模型预测
    :param yolov3_trainer: yolov3检测模型
    :param yolov3_decoder: yolov3模型输出解码器
    :param image_paths: 待预测图片路径列表
    :param save_path：测试结果图形报错路径
    :return:
    """
    import cv2
    logging.info('加载测试数据集：%s', FLAGS.test_label_path)
    input_box_size = np.tile(FLAGS.input_image_size[1::-1], [2])  # 网络输入尺度，(W, H)
    for image_path in image_paths:
        # 读取uint8图片，归一化：[0, 1]的float32 + 原比例resize
        image = tf.constant(cv2.imread(image_path), dtype=tf.int8)
        image = tf.image.resize_image_with_pad(image, target_height=input_box_size[1], target_width=input_box_size[0],
                                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = np.array(image, dtype=np.float)
        predictions = yolov3_trainer.predict(np.expand_dims(image, axis=0))
        [(_, head_8_predicts, head_8_predicts_boxes),
         (_, head_16_predicts, head_16_predicts_boxes),
         (_, head_32_predicts, head_32_predicts_boxes)] = yolov3_decoder.decode(predictions)
        (head_8_prediction, head_8_boxes,
         head_16_prediction, head_16_boxes,
         head_32_prediction, head_32_boxes) = (head_8_predicts[0], head_8_predicts_boxes[0],
                                               head_16_predicts[0], head_16_predicts_boxes[0],
                                               head_32_predicts[0], head_32_predicts_boxes[0])
        # (k, 8)， 归一化尺度->网络输入尺度的[(left top right bottom iou prob class score) ... ]
        high_score_boxes = YOLOv3PostProcessor.filter_boxes(head_8_prediction, head_8_boxes,
                                                            head_16_prediction, head_16_boxes,
                                                            head_32_prediction, head_32_boxes,
                                                            FLAGS.confidence_thresh)
        nms_boxes = YOLOv3PostProcessor.apply_nms(high_score_boxes, FLAGS.nms_thresh)
        in_boxes = YOLOv3PostProcessor.resize_boxes(nms_boxes, target_size=input_box_size)
        image_path = os.path.join(save_path, os.path.basename(image_path))
        YOLOv3PostProcessor.visualize(image, in_boxes, src_box_size=input_box_size, image_path=image_path)
    return


def run():
    # gpu模式
    if FLAGS.gpu_mode != YOLOv3Trainer.CPU_MODE:
        os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.visible_gpu
        # tf.device('/gpu:{}'.format(FLAGS.visible_gpu))
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True  # 按需
        sess = tf.Session(config=config)
        keras.backend.set_session(sess)
    
    generate_logger(filename=FLAGS.log_path)
    logging.info('TensorFlow version: %s', tf.__version__)  # 1.13.1
    logging.info('Keras version: %s', keras.__version__)  # 2.2.4-tf
    
    yolov3_trainer = YOLOv3Trainer()
    
    # 模型训练
    if FLAGS.mode == 'train':
        train(yolov3_trainer)
    # 多GPU模型，需要先转为单GPU模型，然后再执行测试
    elif FLAGS.mode == 'test' or FLAGS.mode == 'predict':
        # 多GPU模型转换为单GPU模型
        if FLAGS.gpu_num > 1:
            yolov3_trainer.convert_multi2single()
            logging.info('多GPU训练模型转换单GPU运行模型成功，请使用单GPU测试！')
            return
        # 进行测试或预测
        yolov3_decoder = YOLOv3Decoder(head_grid_sizes=FLAGS.head_grid_sizes, class_num=FLAGS.class_num,
                                       anchor_boxes=FLAGS.anchor_boxes)
        save_path = FLAGS.save_path
        if save_path is not None:
            if not os.path.isdir(save_path):
                raise ValueError('测试结果图形报错路径不是文件夹！！')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
                
        if FLAGS.mode == 'test':
            test(yolov3_trainer, yolov3_decoder, save_path)
            logging.info('测试结束！！！')
        else:
            images_root_path = FLAGS.image_root_path
            if images_root_path is None or not os.path.isdir(save_path) or save_path is None:
                raise ValueError('待预测图形根目录不存在或不是文件夹！！')
            if save_path is None:
                raise ValueError('预测结果图形报错路径不存在！！')
            image_paths = [os.path.join(images_root_path, file_name)
                           for file_name in os.listdir(images_root_path) if file_name.endswith('.jpg')]
            predict(yolov3_trainer, yolov3_decoder, image_paths, save_path)
            logging.info('预测结果！！！')
    # 将模型保存为pb模型
    elif FLAGS.mode == 'save_pb':
        # 保存模型记得注释eager execution
        yolov3_trainer.save_mobile()
    # 将模型保存为服务器pb模型
    elif FLAGS.mode == 'save_serving':
        # 保存模型记得注释eager execution
        yolov3_trainer.save_serving()
    else:
        raise ValueError('Mode Error!')


if __name__ == '__main__':
    run()
