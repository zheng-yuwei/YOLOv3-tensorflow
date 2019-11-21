# -*- coding: utf-8 -*-
"""
File trainer.py
@author:ZhengYuwei
"""
import os
import logging
import tensorflow as tf
from tensorflow import saved_model
from tensorflow import keras

from configs import FLAGS
from utils.logger_callback import DetailLossLogger
from utils.board_callback import MyTensorBoard
from yolov3.yolov3_detector import YOLOv3Detector
from yolov3.yolov3_loss import YOLOv3Loss


class YOLOv3Trainer(object):
    """
    训练分类器：
    1. 初始化分类器模型、训练参数等；
    2. 调用prepare_data函数准备训练、验证数据集；
    3. 调用train函数训练。
    """

    GPU_MODE = 'gpu'
    CPU_MODE = 'cpu'

    def __init__(self):
        """ 训练初始化 """
        # 构建模型网络
        self.backbone = FLAGS.model_backbone  # 网络类型
        self.input_image_size = FLAGS.input_image_size
        self.head_channel_nums = FLAGS.head_channel_nums

        model = YOLOv3Detector(self.backbone).build(self.input_image_size, self.head_channel_nums,
                                                    FLAGS.head_names)
        # 训练模型: cpu，gpu 或 多gpu
        if FLAGS.gpu_mode == YOLOv3Trainer.GPU_MODE and FLAGS.gpu_num > 1:
            self.model = keras.utils.multi_gpu_model(model, gpus=FLAGS.gpu_num)
        else:
            self.model = model
        self.model.summary()
        self.history = None

        # 加载预训练模型（若有）
        self.checkpoint_path = FLAGS.checkpoint_path
        if self.checkpoint_path is None:
            self.checkpoint_path = 'models/'
        if os.path.isfile(self.checkpoint_path):
            if os.path.exists(self.checkpoint_path):
                self.model.load_weights(self.checkpoint_path)
                logging.info('加载模型成功！')
            else:
                self.checkpoint_path = os.path.dirname(self.checkpoint_path)
        if os.path.isdir(self.checkpoint_path):
            if not os.path.exists(self.checkpoint_path):
                os.makedirs(self.checkpoint_path)
            latest = tf.train.latest_checkpoint(self.checkpoint_path)
            if latest is not None:
                self.model.load_weights(latest)
                logging.info('加载模型成功！')
                logging.info(latest)
        else:
            self.checkpoint_path = os.path.dirname(self.checkpoint_path)
        self.checkpoint_path = os.path.join(self.checkpoint_path, FLAGS.checkpoint_name)

        # 设置模型优化方法
        optimizer = keras.optimizers.SGD(lr=FLAGS.init_lr, momentum=0.95, nesterov=True)
        if FLAGS.optimizer == 'adam':
            optimizer = keras.optimizers.Adam(lr=FLAGS.init_lr, amsgrad=True)  # 用AMSGrad
        elif FLAGS.optimizer == 'radam':
            from utils.radam import RAdam
            optimizer = RAdam(lr=1e-3)
        self.loss_function = YOLOv3Loss(FLAGS.head_grid_sizes, FLAGS.class_num,
                                        FLAGS.anchor_boxes, FLAGS.iou_thresh, FLAGS.loss_weights,
                                        rectified_coord_num=FLAGS.rectified_coord_num,
                                        rectified_loss_weight=FLAGS.rectified_loss_weight,
                                        is_focal_loss=FLAGS.is_focal_loss,
                                        focal_alpha=FLAGS.focal_alpha,
                                        focal_gamma=FLAGS.focal_gamma,
                                        is_tiou_recall=FLAGS.is_tiou_recall).loss
        self.model.compile(optimizer=optimizer, loss=self.loss_function)
        
        # 设置模型训练参数
        self.epoch = FLAGS.epoch

        # 设置训练过程中的回调函数
        cp_callback = keras.callbacks.ModelCheckpoint(self.checkpoint_path, save_weights_only=True,
                                                      verbose=1, period=FLAGS.ckpt_period)
        es_callback = keras.callbacks.EarlyStopping(monitor='loss', min_delta=FLAGS.stop_min_delta,
                                                    patience=FLAGS.stop_patience, verbose=0, mode='min')
        lr_callback = keras.callbacks.LearningRateScheduler(FLAGS.lr_func)
        log_callback = DetailLossLogger(verbose=2)
        tensorboard = MyTensorBoard(log_dir=FLAGS.tensorboard_dir)
        self.callbacks = [tensorboard, cp_callback, es_callback, lr_callback, log_callback]

    def train(self, train_set, val_set, train_steps=FLAGS.steps_per_epoch, val_steps=FLAGS.validation_steps):
        """
        使用训练集和验证集进行模型训练
        :param train_set: 训练数据集的tf.data.Dataset对象
        :param val_set: 验证数据集的tf.data.Dataset对象
        :param train_steps: 每个训练epoch的迭代次数
        :param val_steps: 每个验证epoch的迭代次数
        :return:
        """
        if val_set:
            self.history = self.model.fit(train_set, epochs=self.epoch, validation_data=val_set,
                                          steps_per_epoch=train_steps, validation_steps=val_steps,
                                          callbacks=self.callbacks, verbose=0)
        else:
            self.history = self.model.fit(train_set, epochs=self.epoch, steps_per_epoch=train_steps,
                                          callbacks=self.callbacks, verbose=0)
        logging.info('模型训练完毕！')

    def predict(self, test_images):
        """
        使用测试图片进行模型测试
        :param test_images: 测试图片
        :return:
        """
        predictions = self.model.predict(test_images)
        return predictions

    def convert_multi2single(self):
        """
        将多GPU训练的模型转为单GPU模型，从而可以在单GPU上运行测试
        :return:
        """
        # it's necessary to save the model before use this single GPU model
        multi_model = self.model.layers[FLAGS.gpu_num + 1]  # get single GPU model weights
        dir_name = self.checkpoint_path
        if not os.path.isdir(self.checkpoint_path):
            dir_name = os.path.dirname(self.checkpoint_path)
        latest = tf.train.latest_checkpoint(dir_name)
        save_path = os.path.join(dir_name, 'single_' + os.path.basename(latest))
        multi_model.save_weights(save_path)

    def save_mobile(self):
        """
        保存模型为pb模型：先转为h5，再保存为pb（没法直接转pb）
        """
        # 获取待保存ckpt文件的文件名
        latest = tf.train.latest_checkpoint(os.path.dirname(self.checkpoint_path))
        model_name = os.path.splitext(os.path.basename(latest))[0]
        if not os.path.exists(FLAGS.pb_model_dir):
            os.makedirs(FLAGS.pb_model_dir)
        # 将整个模型保存为h5（包含图结构和参数），然后再重新加载
        h5_path = os.path.join(FLAGS.pb_model_dir, '{}.h5'.format(model_name))
        self.model.save(h5_path, overwrite=True, include_optimizer=False)
        model = keras.models.load_model(h5_path)
        model.summary()
        # 保存pb
        with keras.backend.get_session() as sess:
            output_name = [out.op.name for out in model.outputs]
            input_graph_def = sess.graph.as_graph_def()
            for node in input_graph_def.node:
                node.device = ""
            graph = tf.graph_util.remove_training_nodes(input_graph_def)
            graph_frozen = tf.graph_util.convert_variables_to_constants(sess, graph, output_name)
            tf.train.write_graph(graph_frozen, FLAGS.pb_model_dir, '{}.pb'.format(model_name), as_text=False)
        logging.info("pb模型保存成功！")

    def save_serving(self):
        """
        使用TensorFlow Serving时的保存方式：
            serving-save-dir/
                saved_model.pb
                variables/
                    .data & .index
        """
        outputs = dict()
        for index, name in enumerate(FLAGS.output_name):
            outputs[name] = self.model.outputs[index]

        builder = saved_model.builder.SavedModelBuilder(FLAGS.serving_model_dir)
        signature = saved_model.signature_def_utils.predict_signature_def(inputs={'images': self.model.input},
                                                                          outputs=outputs)
        with keras.backend.get_session() as sess:
            builder.add_meta_graph_and_variables(sess=sess,
                                                 tags=[saved_model.tag_constants.SERVING],
                                                 signature_def_map={'predict': signature})
            builder.save()
        logging.info('serving模型保存成功!')
