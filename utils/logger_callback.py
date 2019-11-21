# -*- coding: utf-8 -*-
"""
File logger_callback.py
@author:ZhengYuwei
"""
import time
import logging
from tensorflow import keras
import tensorflow as tf


class DetailLossLogger(keras.callbacks.Callback):
    """
    训练过程中，详细loss日志打印到logging日志的回调函数
    model.fit() 中 verbose=3 时，该日志按batch打印； verbose=4 时，按epoch打印
    """
    
    def __init__(self, verbose=0):
        super(DetailLossLogger, self).__init__()
        self.verbose = verbose
        self.epochs = 0  # model.fit 中 epochs 入参
        self.seen = 0  # 当前迭代次数
        self.target = 0  # model.fit 中 steps_per_epoch 入参
        self._last_update = 0  # 上一个epoch的结束时间
        self.gamma_regular_loss = []  # BN层gamma正则项损失
        self.gamma_length = 0  # BN层gamma罚项数量
        self.kernel_regular_loss = []  # 卷积层卷积核W正则项损失
        self.kernel_length = 0  # 卷积层卷积核罚项数量
        # 记录各项损失函数
        self.rectified_coord_loss = None
        self.coord_loss_xy = None
        self.coord_loss_wh = None
        self.noobj_iou_loss = None
        self.obj_iou_loss = None
        self.class_loss = None

    def on_train_begin(self, logs=None):
        self.epochs = self.params['epochs']
        self.target = self.params['steps']

        with tf.variable_scope('loss_detail', reuse=True):
            self.rectified_coord_loss = tf.get_variable('rectified_coord_loss')
            self.coord_loss_xy = tf.get_variable('coord_loss_xy')
            self.coord_loss_wh = tf.get_variable('coord_loss_wh')
            self.noobj_iou_loss = tf.get_variable('noobj_iou_loss')
            self.obj_iou_loss = tf.get_variable('obj_iou_loss')
            self.class_loss = tf.get_variable('class_loss')

        for regularization_loss in self.model.losses:
            if regularization_loss.name.find('gamma') != -1:
                self.gamma_regular_loss.append(regularization_loss)
            elif regularization_loss.name.find('kernel') != -1:
                self.kernel_regular_loss.append(regularization_loss)
            else:
                raise ValueError('未知正则化项')
        self.gamma_length = len(self.gamma_regular_loss)
        self.gamma_regular_loss = tf.reduce_sum(self.gamma_regular_loss)
        self.kernel_length = len(self.kernel_regular_loss)
        self.kernel_regular_loss = tf.reduce_sum(self.kernel_regular_loss)

    def on_batch_begin(self, batch, logs=None):
        if self.verbose == 1:
            self._last_update = time.time()

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        # In case of distribution strategy we can potentially run multiple steps
        # at the same time, we should account for that in the `seen` calculation.
        num_steps = logs.get('num_steps', 1)
        self.seen += num_steps
        if self.verbose == 1 and self.seen < self.target:
            self.log(logs)

    def on_epoch_begin(self, epoch, logs=None):
        self.seen = 0
        if self.verbose in (1, 2):
            if self.epochs > 1:
                logging.info('Epoch %d/%d' % (epoch + 1, self.epochs))

        self._last_update = time.time()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if self.verbose in (1, 2):
            self.log(logs)
    
    def log(self, logs=None):
        """
        日志记录，格式：
         - xxxs - lr: 0.001 - loss: 234.35 - regularization_loss: 4.65
         head: /8:
         - rectified_loss: 3.2 - xy_loss: 3.3 - wh_loss: 2.4 - noobj_iou_loss: 4.5 - obj_iou_loss: 2.4 - cls_loss: 4.5
         head: /16:
         ...
         head: /32:
         ...
        :param logs:
        :return:
        """
        now = time.time()
        info = '\n - %.0fs' % (now - self._last_update)
        
        log_values = [('lr', logs['lr'])]
        for k in self.params['metrics']:
            if k in logs:
                log_values.append((k, logs[k]))
        
        log_values.append(('gamma_regular_loss({})'.format(self.gamma_length),
                           keras.backend.eval(self.gamma_regular_loss)))
        log_values.append(('kernel_regular_loss({})'.format(self.kernel_length),
                           keras.backend.eval(self.kernel_regular_loss)))
        log_values.append((' ', '\n'))
        
        rectified_coord_loss = keras.backend.eval(self.rectified_coord_loss)
        coord_loss_xy = keras.backend.eval(self.coord_loss_xy)
        coord_loss_wh = keras.backend.eval(self.coord_loss_wh)
        noobj_iou_loss = keras.backend.eval(self.noobj_iou_loss)
        obj_iou_loss = keras.backend.eval(self.obj_iou_loss)
        class_loss = keras.backend.eval(self.class_loss)
        
        for i, head in enumerate(('/8:\n', '/16:\n', '/32:\n')):
            log_values.append(('head', head))
            log_values.append(('rectified_loss', rectified_coord_loss[i]))
            log_values.append(('xy_loss', coord_loss_xy[i]))
            log_values.append(('wh_loss', coord_loss_wh[i]))
            log_values.append(('noobj_iou_loss', noobj_iou_loss[i]))
            log_values.append(('obj_iou_loss', obj_iou_loss[i]))
            log_values.append(('cls_loss', class_loss[i]))
            log_values.append((' ', '\n'))
        
        for (key, value) in log_values:
            if isinstance(value, str):
                info += ' - %s: %s' % (key, value)
            elif value > 1e-3:
                info += ' - %s: %.4f' % (key, value)
            else:
                info += ' - %s: %.4e' % (key, value)
        
        logging.info(info)
        return
