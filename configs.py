# -*- coding: utf-8 -*-
"""
File configs.py
@author:ZhengYuwei
"""
import datetime
import numpy as np
from easydict import EasyDict
from yolov3.yolov3_detector import YOLOv3Detector

FLAGS = EasyDict()

# check_*是在项目前期定学习率时用于测试的，后续调整train_*学习率就行
FLAGS.check_step_epoch = np.array([2, 4, 6, 8, 10, 12, 14], np.int)
FLAGS.check_step_lr = np.array([0.00001, 0.0001, 0.001, 0.01, 0.1, 1., 10.0], dtype=np.float) * 1e-3
FLAGS.train_step_epoch = np.array([20, 60, 80, 220, 260, 280, 300], np.int)
FLAGS.train_step_lr = np.array([0.01, 1., 0.1, 1., 0.1, 0.01, 0.001], dtype=np.float) * 1e-3
# 训练时期或是调整学习率时期
FLAGS.step_epoch = FLAGS.train_step_epoch
FLAGS.step_lr = FLAGS.train_step_lr


def lr_func(epoch):
    i = 0
    while i < len(FLAGS.step_epoch) and epoch > FLAGS.step_epoch[i]:
        i += 1
    return FLAGS.step_lr[i]


# 数据集
FLAGS.train_set_dir = 'dataset/test_sample/images'
FLAGS.train_label_path = 'dataset/test_sample/label.txt'
FLAGS.test_set_dir = 'dataset/test_sample/images'
FLAGS.test_label_path = 'dataset/test_sample/label.txt'
# 模型权重的L2正则化权重直接写在对应模型的骨干网络定义文件中
FLAGS.input_image_size = np.array([384, 480, 3], dtype=np.int)  # [H, W, C]
FLAGS.anchor_boxes = [[(0.06618181818181816, 0.1025177510694752), (0.18544278606965178, 0.13160367921287464),
                      (0.13, 0.32733333333333337)],
                      [(0.13, 0.32733333333333337), (0.303806787732042, 0.34370030784316496)],
                      [(0.303806787732042, 0.34370030784316496), (0.4667050847457627, 0.5281262429095761),
                      (0.7906945888923907, 0.7888860433597275)]]  # [W, H]，分别是head_8, head_16, head_32的anchor boxes
FLAGS.class_num = 0
FLAGS.box_num = np.array([len(anchor_boxes) for anchor_boxes in FLAGS.anchor_boxes], dtype=np.int)
FLAGS.box_len = 4 + 1 + FLAGS.class_num
FLAGS.head_channel_nums = FLAGS.box_num * FLAGS.box_len
FLAGS.head_grid_sizes = [np.divide(FLAGS.input_image_size[0:2], 8).astype(np.int),
                         np.divide(FLAGS.input_image_size[0:2], 16).astype(np.int),
                         np.divide(FLAGS.input_image_size[0:2], 32).astype(np.int)]  # [H, W],
FLAGS.head_names = ['yolov3_head_8', 'yolov3_head_16', 'yolov3_head_32', ]
FLAGS.iou_thresh = 0.8  # 大于该IOU阈值，不计算该anchor的背景IOU误差
# 不同head(/8, /16, /32)不同损失项的权：[coord_xy, coord_wh, noobj, obj, cls_prob]，蓝，绿，红
FLAGS.loss_weights = [(5, 5, 0.05, 3, 1), (8, 8, 0.05, 2, 1), (10, 10, 0.05, 2, 1)]
# 训练参数
FLAGS.train_set_size = 20
FLAGS.val_set_size = 20
FLAGS.batch_size = 3
# 若你已经有预训练模型，给rectified_coord_num赋值为-1即可
FLAGS.rectified_coord_num = 1464  # 前期给坐标做矫正损失的图片数，源代码 12800，train-from-scratch需要用
FLAGS.rectified_loss_weight = [1.0, 1.0, 1.0]  # 前期矫正坐标的损失的权重，源代码 0.01，可调，太大的话coord_loss_wh会跟着爆炸
FLAGS.epoch = 300
FLAGS.init_lr = 0.0002  # nadam推荐使用值
# 训练参数
FLAGS.mode = 'test'  # train, test, predict, save_pb, save_serving
FLAGS.model_backbone = YOLOv3Detector.BACKBONE_RESNET_18
FLAGS.optimizer = 'radam'  # sgdm, adam, radam
FLAGS.is_augment = True
FLAGS.is_label_smoothing = False
FLAGS.is_focal_loss = False
FLAGS.focal_alpha = 1.0
FLAGS.focal_gamma = 2.0
FLAGS.is_gradient_harmonized = False
FLAGS.is_tiou_recall = False
FLAGS.type = FLAGS.model_backbone + '-' + FLAGS.optimizer
FLAGS.type += ('-aug' if FLAGS.is_augment else '')
FLAGS.type += ('-smooth' if FLAGS.is_label_smoothing else '')
FLAGS.type += ('-focal' if FLAGS.is_focal_loss else '')
FLAGS.type += ('-ghm' if FLAGS.is_gradient_harmonized else '')
FLAGS.type += ('-TIOU' if FLAGS.is_tiou_recall else '')
FLAGS.log_path = 'logs/log-{}.txt'.format(FLAGS.type)
# 训练参数
FLAGS.steps_per_epoch = int(np.ceil(FLAGS.train_set_size / FLAGS.batch_size))
FLAGS.validation_steps = int(np.ceil(FLAGS.val_set_size / FLAGS.batch_size))
# callback的参数
FLAGS.ckpt_period = 50  # 模型保存
FLAGS.stop_patience = 500  # early stop
FLAGS.stop_min_delta = 0.0001
FLAGS.lr_func = lr_func  # 学习率更新函数
# tensorboard日志保存目录
FLAGS.root_path = ''  # /cache/zhengyuwei/license-plate-recognition/
FLAGS.tensorboard_dir = FLAGS.root_path + 'logs/' + \
                        'lpr-{}-{}'.format(FLAGS.type, datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
# 模型保存
FLAGS.checkpoint_path = FLAGS.root_path + 'models/{}/'.format(FLAGS.type)
FLAGS.checkpoint_name = 'lp-recognition-{}'.format(FLAGS.type) + '-{epoch: 3d}-{loss: .5f}.ckpt'
FLAGS.serving_model_dir = FLAGS.root_path + 'models/serving'
FLAGS.pb_model_dir = FLAGS.root_path + 'models/pb'

# 测试参数
FLAGS.confidence_thresh = 0.8  # 基础置信度
FLAGS.nms_thresh = 0.4  # nms阈值
FLAGS.save_path = 'dataset/test_result/'  # 测试结果图形报错路径
FLAGS.image_root_path = None  # 预测图片的根目录
# 训练gpu
FLAGS.gpu_mode = 'gpu'
FLAGS.gpu_num = 1
FLAGS.visible_gpu = '0'  # ','.join([str(_) for _ in range(FLAGS.gpu_num)])
