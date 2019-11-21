# -*- coding: utf-8 -*-
"""
File yolov3_detector.py
@author:ZhengYuwei
"""
import logging
from tensorflow import keras
from backbone.resnet18 import ResNet18
from backbone.resnet18_v2 import ResNet18_v2
from backbone.resnext import ResNeXt18
from backbone.mixnet18 import MixNet18
from backbone.mobilenet_v2 import MobileNetV2


class YOLOv3Detector(object):
    """
    检测器，自定义了YOLOv3的head
    """
    BACKBONE_RESNET_18 = 'resnet-18'
    BACKBONE_RESNET_18_V2 = 'resnet-18-v2'
    BACKBONE_RESNEXT_18 = 'resnext-18'
    BACKBONE_MIXNET_18 = 'mixnet-18'
    BACKBONE_MOBILENET_V2 = 'mobilenet-v2'
    BACKBONE_TYPE = {
        BACKBONE_RESNET_18: ResNet18,
        BACKBONE_RESNET_18_V2: ResNet18_v2,
        BACKBONE_RESNEXT_18: ResNeXt18,
        BACKBONE_MOBILENET_V2: MobileNetV2,
        BACKBONE_MIXNET_18: MixNet18
    }
    
    def __init__(self, backbone_name):
        """
        YOLO v3检测模型初始化
        :param backbone_name: 骨干网络名称, BACKBONE_TYPE.keys()
        """
        logging.info('构造YOLOv3模型，基础网络：%s', backbone_name)
        self.backbone_name = backbone_name
        if backbone_name in self.BACKBONE_TYPE.keys():
            self.backbone = self.BACKBONE_TYPE[backbone_name]
        else:
            raise ValueError("没有该类型的基础网络！")

    def build(self, input_image_size, head_channel_nums, head_names):
        """
        构建全卷积网络backbone基础网络的YOLOv3 keras.models.Model对象
        :param input_image_size: 输入尺寸
        :param head_channel_nums: 检测头channel数，grid_shape = 5 * （4 + 1 + class_num）
        :param head_names: 检测头的名字
        :return: 全卷积网络的YOLOv3 keras.models.Model对象
        """
        if len(input_image_size) != 3:
            raise Exception('模型输入形状必须是3维形式')
    
        input_x = keras.layers.Input(shape=input_image_size)
        backbone_output = self.backbone.build(input_x)
        outputs = self._detection_head(backbone_output, head_channel_nums, head_names)
        model = keras.models.Model(inputs=input_x, outputs=outputs, name=self.backbone_name)
        return model

    def _detection_head(self, nets, head_channel_nums, head_names):
        """
        YOLOv3的head，上接全卷积网络backbone，输出(/8, /16, /32)尺度的head_channel_nums个channel的矩阵
        :param nets: 全卷积网络backbone的3个尺度的输出
        :param head_channel_nums: 3个检测头channel数
        :param head_names: 3个检测头的名字
        :return: (/8, /16, /32)尺度输出矩阵，经过reshape后所形成，
                (N, H/32, W/32, head_8_channel*16 + head_16_channel*4 + head_32_channel)矩阵
        """
        sub_stride_8_net, sub_stride_16_net, sub_stride_32_net = nets
        stride_8_channel_num, stride_16_channel_num, stride_32_channel_num = head_channel_nums
        stride_8_head_name, stride_16_head_name, stride_32_head_name = head_names
        
        head_32_feature = self._yolov3_stride_32_head(sub_stride_32_net, stride_32_channel_num, stride_32_head_name)
        merge_net, head_16_feature = self._yolov3_stride_16_head(sub_stride_32_net, sub_stride_16_net,
                                                                 stride_16_channel_num, stride_16_head_name)
        head_8_feature = self._yolov3_stride_8_head(merge_net, sub_stride_8_net,
                                                    stride_8_channel_num, stride_8_head_name)
        # 把三个输出reshape到stride 32的尺寸，concat一起输出，这样才能成为一个loss function的输入；或者使用add_loss实现
        batch_size, head_32_height, head_32_width, _ = keras.backend.int_shape(head_32_feature)
        reshape_layer_op = keras.layers.Reshape(target_shape=[head_32_height, head_32_width, -1])
        reshape_head_16_feature = reshape_layer_op(head_16_feature)
        reshape_head_8_feature = reshape_layer_op(head_8_feature)
        merge_head = keras.layers.concatenate([reshape_head_8_feature, reshape_head_16_feature, head_32_feature],
                                              axis=self.backbone.CHANNEL_AXIS)
        return merge_head
    
    def _yolov3_stride_32_head(self, sub_stride_32_net, stride_32_channel_num, stride_32_head_name):
        """
        YOLOv3尺度为 /32 的head，接(3,3), (1,1)两层卷积
        :param sub_stride_32_net: 骨干网络中stride 32的输出特征
        :param stride_32_channel_num: stride 32的head的输出通道数
        :param stride_32_head_name: stride 32的head的名称
        :return: 输出(N, H/32, W/32, stride_32_channel_num)矩阵
        """
        net = self.backbone.conv_bn(sub_stride_32_net, 512)   # darknet-53是1024
        net = self.backbone.activation(net)
        output = keras.layers.Conv2D(filters=stride_32_channel_num, kernel_size=(1, 1),
                                     kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
                                     activation=None, use_bias=True, name=stride_32_head_name)(net)
        return output

    def _yolov3_stride_16_head(self, stride_32_feature, sub_stride_16_net, stride_16_channel_num, stride_16_head_name):
        """
        YOLOv3尺度为 /32 的head
        :param stride_32_feature: stride 32的特征图，用来与stride 16的特征做多尺度特征融合
        :param sub_stride_16_net: 骨干网络中stride 16的输出特征
        :param stride_16_channel_num: stride 16的head的输出通道数
        :param stride_16_head_name: stride 16的head的名称
        :return: 输出(N, H/16, W/16, stride_16_channel_num)矩阵
        """
        # stride_32 -> conv -> up_sample -> concat(stride_16)
        net = self.backbone.conv_bn(stride_32_feature, filters=256, strides=(1, 1))
        net = self.backbone.activation(net)
        up_sample_net = keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest')(net)
        merge_net = keras.layers.concatenate([up_sample_net, sub_stride_16_net], axis=-1)
        # conv(1, 1)
        merge_net = self.backbone.conv_bn(merge_net, filters=256, kernel_size=(1, 1))
        merge_net = self.backbone.activation(merge_net)
        # conv(3, 3) -> detection conv
        net = self.backbone.conv_bn(merge_net, filters=512, kernel_size=(3, 3))
        net = self.backbone.activation(net)
        output = keras.layers.Conv2D(filters=stride_16_channel_num, kernel_size=(1, 1),
                                     kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
                                     activation=None, use_bias=True, name=stride_16_head_name)(net)
        return merge_net, output

    def _yolov3_stride_8_head(self, stride_16_feature, sub_stride_8_net, stride_8_channel_num, stride_8_head_name):
        """
        YOLOv3尺度为 /8 的head
        :param stride_16_feature: stride 16的特征图，用来与stride 8的特征做多尺度特征融合
        :param sub_stride_8_net: 骨干网络中stride 8的输出特征
        :param stride_8_channel_num: stride 8的head的输出通道数
        :param stride_8_head_name: stride 8的head的名称
        :return: 输出(N, H/8, W/8, stride_8_channel_num)矩阵
        """
        # stride_16 -> conv -> up_sample -> concat(stride_8)
        net = self.backbone.conv_bn(stride_16_feature, filters=128, kernel_size=(1, 1))
        net = self.backbone.activation(net)
        up_sample_net = keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest')(net)
        merge_net = keras.layers.concatenate([up_sample_net, sub_stride_8_net], axis=-1)
        # conv(1, 1)
        merge_net = self.backbone.conv_bn(merge_net, filters=128, kernel_size=(1, 1))
        merge_net = self.backbone.activation(merge_net)
        # conv(3, 3) -> detection conv
        merge_net = self.backbone.conv_bn(merge_net, filters=256, kernel_size=(3, 3))
        merge_net = self.backbone.activation(merge_net)
        output = keras.layers.Conv2D(filters=stride_8_channel_num, kernel_size=(1, 1),
                                     kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
                                     activation=None, use_bias=True, name=stride_8_head_name)(merge_net)
        return output
    

if __name__ == '__main__':
    """
    可视化网络结构，使用plot_model需要先用conda安装GraphViz、pydotplus
    """
    for test_backbone_name in YOLOv3Detector.BACKBONE_TYPE.keys():
        yolo_v3 = YOLOv3Detector(test_backbone_name)
        test_model = yolo_v3.build(input_image_size=(384, 480, 3), head_channel_nums=[3 * (4 + 1 + 20)]*3,
                                   head_names=['yolov3_head_8', 'yolov3_head_16', 'yolov3_head_32', ])
        keras.utils.plot_model(test_model, to_file='../images/{}.svg'.format(test_backbone_name), show_shapes=True)
        print('backbone: ', test_backbone_name)
        test_model.summary()
        print('=====================' * 5)
