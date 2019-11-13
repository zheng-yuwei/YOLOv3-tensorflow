# -*- coding: utf-8 -*-
"""
File logger_callback.py
@author:ZhengYuwei
"""
from tensorflow import keras
import tensorflow as tf


class DetailLossProgbarLogger(keras.callbacks.ProgbarLogger):
    """ 训练过程中，每N个batch打印log到stdout的回调函数 """
    def __init__(self, count_mode='samples', stateful_metrics=None):
        super(DetailLossProgbarLogger, self).__init__(count_mode, stateful_metrics)
    
    def on_epoch_end(self, batch, logs=None):
        logs = logs or {}
        for k in self.params['metrics']:
            if k in logs:
                self.log_values.append((k, logs[k]))
                
        with tf.variable_scope('loss_detail', reuse=True):
            self.log_values.append(('rectified_coord_loss',
                                    keras.backend.eval(tf.get_variable('rectified_coord_loss'))))
            self.log_values.append(('coord_loss_xy',
                                    keras.backend.eval(tf.get_variable('coord_loss_xy'))))
            self.log_values.append(('coord_loss_wh',
                                    keras.backend.eval(tf.get_variable('coord_loss_wh'))))
            self.log_values.append(('noobj_iou_loss',
                                    keras.backend.eval(tf.get_variable('noobj_iou_loss'))))
            self.log_values.append(('obj_iou_loss',
                                    keras.backend.eval(tf.get_variable('obj_iou_loss'))))
            self.log_values.append(('class_loss',
                                    keras.backend.eval(tf.get_variable('class_loss'))))
            self.log_values.append(('regularization_loss',
                                    keras.backend.eval(tf.reduce_sum(self.model.losses))))
        self.progbar.update(self.seen, self.log_values)
