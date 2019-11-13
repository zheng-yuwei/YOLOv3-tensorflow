# -*- coding: utf-8 -*-
"""
File script_note.py
@author:ZhengYuwei
"""
import tensorflow as tf


def visual_meta_with_tensorboard():
    """ 使用tensorflow查看checkpoint、meta文件中的网络结构 """
    sess = tf.Session()
    saver = tf.train.import_meta_graph('model.ckpt.meta')  # load meta
    saver.restore(sess, 'model.ckpt')  # load ckpt
    writer = tf.summary.FileWriter(logdir='logs', graph=tf.get_default_graph())  # write to event
    writer.flush()
    return
