"""
数据处理和补丁添加
Loss计算  # TGJ
"""

import fnmatch
import math
import os
# from random 
import random
import sys
import time
import gc
from tokenize import single_quoted
import numpy as np
from numpy.core.shape_base import block
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from darknet_v3 import Darknet
import utils
from median_pool import MedianPool2d
# import edge_extractor_oneStage

ANCHOR_PATH = "data/yolov3_anchors.txt"
DOTA_NAMES = "data/dota.names"
SCALE_FACTOR = 2.


def read_image(path):
    """
    读取一个已经训练好的补丁
    Read an input image to be used as a patch

    :param path: Path to the image to be read.
    :return: Returns the transformed patch as a pytorch Tensor.
    """
    patch_img = Image.open(path).convert('RGB')
    tf = transforms.ToTensor()

    adv_patch_cpu = tf(patch_img)
    return adv_patch_cpu

# def get_boxes_loss(output, anchors, num_anchors):
# torch.nn.MSEloss 有除以N


def bbox_reg(bbox_extractor):
    attack_bbox = torch.tensor([1e-6, 1e-6, 1e-6, 1e-6]).cuda()
    bbox_mse = []
    for box in bbox_extractor:
        # print(len(box))
        box_mse = torch.nn.MSELoss()(attack_bbox, box) * len(box)
        bbox_mse.append(box_mse)
    return bbox_mse


def bbox_decode(output, num_classes, anchors, num_anchors):
    '''   batch x 60 x 19 x 19
              batch x 60 x 38 x 38
              batch x 60 x 76 x 76
    '''

    img_size = (608, 608)
    batch = output.size(0)
    h = output.size(2)
    w = output.size(3)
    stride_h = img_size[1] / h
    stride_w = img_size[0] / w

    scaled_anchors = [(anchor_width / stride_w, anchor_height / stride_h)
                      for anchor_width, anchor_height in anchors]
    #   这里的anchors本来是全局的
    output = output.view(batch*num_anchors, 5+num_classes,
                         h*w)  # batch*3, 20, 19*19
    output = output.transpose(0, 1).contiguous()  # 20, batch*3, 19*19
    output = output.view(5+num_classes, batch *
                         num_anchors*h*w)  # 20, batch*3* 19*19
    grid_x = torch.linspace(0, w-1, w).repeat(h, 1).repeat(batch *
                                                           num_anchors, 1, 1).view(batch*num_anchors*h*w).cuda()
    grid_y = torch.linspace(0, h-1, h).repeat(w, 1).t().repeat(batch *
                                                               num_anchors, 1, 1).view(batch*num_anchors*h*w).cuda()
    xs = torch.sigmoid(output[0]) + grid_x
    ys = torch.sigmoid(output[1]) + grid_y

    anchor_w = torch.Tensor(scaled_anchors).index_select(
        1, torch.LongTensor([0]))
    anchor_h = torch.Tensor(scaled_anchors).index_select(
        1, torch.LongTensor([1]))
    anchor_w = anchor_w.repeat(batch, 1).repeat(
        1, 1, h*w).view(batch*num_anchors*h*w).cuda()
    anchor_h = anchor_h.repeat(batch, 1).repeat(
        1, 1, h*w).view(batch*num_anchors*h*w).cuda()

    ws = torch.exp(output[2]) * anchor_w
    hs = torch.exp(output[3]) * anchor_h

    # xs = xs * stride_w
    # ys = ys * stride_h
    # ws = ws * stride_w
    # hs = hs * stride_h

    xs = xs / w
    ys = ys / h
    ws = ws / w
    hs = hs / h  # 这里可进行归一化处理
    output[0] = xs
    output[1] = ys
    output[2] = ws
    output[3] = hs  # 20, batch*3, 19*19

    output = output.view(5+num_classes, batch*num_anchors,
                         h*w)  # 20, batch*3, 19*19
    output = output.transpose(0, 1).contiguous()  # batch*3, 20, 19*19
    output = output.view(batch, num_anchors*(5+num_classes), h, w)

    return output


class MaxProbExtractor(nn.Module):
    """此部分计算YOLO的输出置信度。
    MaxProbExtractor: extracts max class probability for class from YOLO output.
    提取的是输出向量中最大的概率类别，因为最后是要进行优化，使得分类类别不是person
    Module providing the functionality necessary to extract the max class probability for one class from YOLO output.

    """

    def __init__(self, cls_id, num_cls, config):
        """
        cls_id：给定的分类结果id，默认为0，
        num_cls：表示分类数量，在train_patch中调用
        """
        super(MaxProbExtractor, self).__init__()
        self.cls_id = cls_id  # 传入参数
        self.num_cls = num_cls  # 传入参数
        self.config = config

        #-----------------------------------------------#
        #   2021-08-25学习了YOLO-v3后更新
        #   1. 上面这里在取normal_confs的时候，使用的是softmax处理，但是
        #   softmax会将其映射为一个概率，即\sum()=1，这样的话其实所有的值
        #   都变小了
        #   因此将其改成sigmoid是不是会好点？（没必要映射为softmax吧？）
        #   2. 本函数是取出了某一个类中最大的概率作为优化函数，即如果是person
        #   目标，图片中有n个person，也是取其中最大的那个person，然后随着
        #   优化的迭代，会使得所有的person概率都较小
        #   这样理解也可以，但是如果一次性选择所有的person概率，而不是max，
        #   是否能提高效率？
        #   但是可能会有很多检测框都是预测为person，这样能不能先使用thresh进行筛选？
        #   3. 那更夸张直接的，能不能直接使用模型的输出，而不进行sigmoid或soft处理？
        #-----------------------------------------------#

    # def forward(self, YOLOoutput, anchors, num_anchors):

    def forward(self, YOLOoutputs, sigmoid_mode=False):
        #   因为这里不需要筛选，只需要对[x,y,w,h,obj_conf,{classes_conf}]
        #   进行解耦，因此也简单，直接先concatenate
        # attack_bbox = torch.tensor([1e-6,1e-6,1e-6,1e-6])
        #   YOLOoutput：
        '''   batch x 60 x 19 x 19
              batch x 60 x 38 x 38
              batch x 60 x 76 x 76
        '''
        #   '''anchors_step = len(anchors) // num_anchors'''
        anchors = utils.get_anchors(ANCHOR_PATH)
        num_anchors = len(anchors)
        class_names = utils.load_class_names(DOTA_NAMES)
        num_classes = len(class_names)

        output_single_dim = []
        for i, output in enumerate(YOLOoutputs):
            batch = output.size(0)

            h = output.size(2)
            w = output.size(3)  # 32 x 32

            # output = output.view(
            # batch, 5, 5 + self.num_cls, h * w)  #   [batch, num_anchors, x, y, z]
            output = bbox_decode(output, num_classes, anchors[i], num_anchors)
            #   [batch, 3*(5+15), 19 ,19]
            #   [batch, 3*(5+15), 38 ,38]
            #   [batch, 3*(5+15), 76 ,76]
            output = output.view(batch, 3, 5 + self.num_cls, h * w)
            # [batch, 20, 3, 19*19]
            output = output.transpose(1, 2).contiguous()
            output = output.view(batch, 5 + self.num_cls,
                                 3 * h * w)  # [batch, 20, x]
            output_single_dim.append(output)
        #---------------------------------------------------------------------#
        # 以上部分不变
        #---------------------------------------------------------------------#
        # output_cat：[batch, 20, 22743]
        output_cat = torch.cat(output_single_dim, 2)
        #   使用torch.cat()将来list合成成了tensor？？

        # 这部分代码实现的是使用sigmoid激活后loss

        # [batch, 3*(19*19+38*38+78*78)]
        if sigmoid_mode:
            output_objectness = torch.sigmoid(output_cat[:, 4, :])
            max_obj_conf, _ = torch.max(output_objectness, dim=1)  # 这用来确定权重参数
            output_cls_conf = output_cat[:, 5:5 +
                                         self.num_cls, :]  # [batch, 15, 22743]
            # 对第二维进行softmax操作, 对类别概率进行softmax激活
            normal_confs = torch.sigmoid(output_cls_conf)
            confs_for_class = normal_confs[:, self.cls_id, :]  # [batch, 22743]
            max_cls_conf, _ = torch.max(confs_for_class, dim=1)  # 这用来确定权重参数

            # confs_if_object = self.config.loss_target(
            #     output_objectness, confs_for_class)  # 最终使用的loss在这里确定[batch, 22743]

            # max_det_conf, max_conf_idx = torch.max(
            #     confs_if_object, dim=1)  # 这个地方应不应该只取最大值还值得商榷

        #   这部分提取的是模型的原始输出，即未经过sigmoid激活部分。
        else:
            output_objectness_raw = output_cat[:, 4, :]
            output_cls = output_cat[:, 5:5 + self.num_cls, :]
            confs_for_class_raw = output_cls[:, self.cls_id, :]
            # confs_if_object_raw = self.config.loss_target(
            #     output_objectness_raw, confs_for_class_raw)  # 最终使用的loss在这里确定[batch, 22743]
            max_cls_conf, _ = torch.max(confs_for_class_raw, dim=1)
            max_obj_conf, _ = torch.max(output_objectness_raw, dim=1)

        #--------------------------------------------------------------#
        #   以下使用target输出，而不是max输出
        #--------------------------------------------------------------#

        #   原始输出作为优化loss
        '''
        output_objectness_raw = output_cat[:, 4, :]
        zero_object = torch.zeros_like(output_objectness_raw)   #   筛选
        output_objectness_raw = torch.where(output_objectness_raw<0., zero_object, output_objectness_raw)
        # temp_obj_sum = torch.sum(output_objectness_raw, dim = 1)
        output_cls = output_cat[:, 5:5 + self.num_cls, :]
        confs_for_class_raw = output_cls[:, self.cls_id, :]
        zero_class = torch.zeros_like(confs_for_class_raw)
        confs_for_class_raw = torch.where(confs_for_class_raw<0., zero_class, confs_for_class_raw)
        # temp_confs_for_class = torch.sum(confs_for_class_raw, dim = 1)
        confs_if_object_raw = self.config.loss_target(
            output_objectness_raw, confs_for_class_raw)  # 最终使用的loss在这里确定[batch, 22743]
        max_conf = torch.sum(confs_if_object_raw, dim=1)
        '''
        '''
        #   sigmoid激活后作为优化loss
        # [batch, 3*(19*19+38*38+78*78)]
        output_objectness = torch.sigmoid(output_cat[:, 4, :])
        zero_object = torch.zeros_like(output_objectness)
        output_objectness = torch.where(output_objectness<0.4, zero_object, output_objectness)
        # output_objectness[output_objectness < 0.4] = 0  #   这一行虽然简单，但是不适用反向传播
        temp_obj_sum = torch.sum(output_objectness, dim = 1)
        output_cls_conf = output_cat[:, 5:5 +
                                     self.num_cls, :]  # [batch, 15, 22743]
        normal_confs = torch.sigmoid(output_cls_conf)
        confs_for_class = normal_confs[:, self.cls_id, :]  # [batch, 22743]
        zero_class = torch.zeros_like(confs_for_class)
        confs_for_class = torch.where(confs_for_class<0.4, zero_class, confs_for_class)
        temp_confs_for_class = torch.sum(confs_for_class, dim = 1)
        confs_if_object = self.config.loss_target(
            output_objectness, confs_for_class)  #
        max_conf = torch.sum(confs_if_object, dim=1)
        '''

        #--------------------------------------------------------------#
        #   以下仅使用obj_wo_sigmoid作为loss项，并增加bb_box-MSE_loss
        #--------------------------------------------------------------#
        '''
        output_objectness_raw = output_cat[:, 4, :]
        output_cls = output_cat[:, 5:5 + self.num_cls, :]
        # max_obj_conf, _ = torch.max(output_objectness_raw, dim=1)  #   这用来确定权重参数
        confs_for_class_raw = output_cls[:, self.cls_id, :]
        # max_cls_conf, _ = torch.max(confs_for_class_raw, dim=1)
        confs_if_object_raw = self.config.loss_target(
            output_objectness_raw, confs_for_class_raw)  # 最终使用的loss在这里确定[batch, 22743]
        max_conf, max_conf_idx = torch.max(
            confs_if_object_raw, dim=1)
        '''
        '''output_objectness = torch.sigmoid(output_cat[:, 4, :])  # [batch, 3*(19*19+38*38+78*78)]
        output_cls_conf = output_cat[:, 5:5 + self.num_cls, :]  # [batch, 15, 22743]
        normal_confs = torch.sigmoid(output_cls_conf)  # 对第二维进行softmax操作, 对类别概率进行softmax激活
        confs_for_class = normal_confs[:, self.cls_id, :]  # [batch, 22743]

        confs_if_object = self.config.loss_target(
            output_objectness, confs_for_class)  # 最终使用的loss在这里确定[batch, 22743]
        max_conf, max_conf_idx = torch.max(
            confs_if_object, dim=1) '''
        #   增加回归框loss

        # xs = output_cat[:, 0, :]
        # ys = output_cat[:, 1, :]
        # ws = output_cat[:, 2, :]
        # hs = output_cat[:, 3, :]
        # # print("min xs : {}, min ys : {}, min ws : {}, min hs : {}".format(torch.min(
        # #     xs, dim=1), torch.min(ys, dim=1), torch.min(ws, dim=1), torch.min(hs, dim=1)))

        # bbox_extrac = []
        # for i in range(len(max_conf_idx)):
        #     index = max_conf_idx[i]
        #     bbox = torch.Tensor(
        #         [xs[i, index], ys[i, index], ws[i, index], hs[i, index]]).cuda()
        #     bbox_extrac.append(bbox)

        # bbox_extract_loss = bbox_reg(bbox_extrac)  # 提取目标检测框的reg_loss
        # bbox_extract_loss_ = torch.Tensor(bbox_extract_loss)

        return max_obj_conf, max_cls_conf  #

        '''
        obj_confs = []
        cls_confs = []
        for i in range(len(max_conf_idx)):

            obj_confs.append(output_objectness[i, max_conf_idx[i]])
            cls_confs.append(confs_for_class[i, max_conf_idx[i]])
        '''

        '''
        bbox_extrac = []
        for i in range(len(max_conf_idx)):
            index = max_conf_idx[i]
            bbox = torch.Tensor([xs[i, index]/w,ys[i, index]/h,ws[i, index]/w,hs[i, index]/h]).cuda()
            bbox_extrac.append(bbox)

        bbox_extract_loss = bbox_reg(bbox_extrac)   #   提取目标检测框的reg_loss
        bbox_extract_loss_ = torch.Tensor(bbox_extract_loss)
        '''

        '''
        obj_confs = torch.Tensor(obj_confs)
        cls_confs = torch.Tensor(cls_confs)  # list都要转tensor
  
        '''


class NPSCalculator(nn.Module):
    """NPSCalculator: calculates the non-printability score of a patch.

    Module providing the functionality necessary to calculate the non-printability score (NMS) of an adversarial patch.

    """

    def __init__(self, printability_file, patch_side):
        super(NPSCalculator, self).__init__()
        self.printability_array = nn.Parameter(self.get_printability_array(
            printability_file, patch_side), requires_grad=False)
        # 获得可打印分数数组，可参考"adv_patch.py"理解
        #  nn.Parameter函数？将一个不可训练的类型转换称可训练的类型

    def forward(self, adv_patch):
        # calculate euclidian distance between colors in patch and colors in printability_array
        # square root of sum of squared difference
        color_dist = (adv_patch - self.printability_array + 0.000001)
        color_dist = color_dist ** 2
        color_dist = torch.sum(color_dist, 1) + 0.000001
        color_dist = torch.sqrt(color_dist)
        # only work with the min distance
        # test: change prod for min (find distance to closest color)
        color_dist_prod = torch.min(color_dist, 0)[0]
        # calculate the nps by summing over all pixels
        nps_score = torch.sum(color_dist_prod, 0)
        nps_score = torch.sum(nps_score, 0)
        return nps_score / torch.numel(adv_patch)

    def get_printability_array(self, printability_file, side):
        printability_list = []

        # read in printability triplets and put them in a list
        with open(printability_file) as f:
            for line in f:
                printability_list.append(line.split(","))

        printability_array = []
        for printability_triplet in printability_list:
            printability_imgs = []
            red, green, blue = printability_triplet
            printability_imgs.append(np.full((side, side), red))
            printability_imgs.append(np.full((side, side), green))
            printability_imgs.append(np.full((side, side), blue))
            printability_array.append(printability_imgs)

        printability_array = np.asarray(printability_array)
        printability_array = np.float32(printability_array)
        pa = torch.from_numpy(printability_array)
        return pa


class TotalVariation(nn.Module):
    """TotalVariation: calculates the total variation of a patch.

    Module providing the functionality necessary to calculate the total vatiation (TV) of an adversarial patch.

    """

    def __init__(self):
        super(TotalVariation, self).__init__()

    def forward(self, adv_patch):
        # bereken de total variation van de adv_patch
        tvcomp1 = torch.sum(torch.abs(
            adv_patch[:, :, 1:] - adv_patch[:, :, :-1] + 0.000001), 0)
        tvcomp1 = torch.sum(torch.sum(tvcomp1, 0), 0)
        tvcomp2 = torch.sum(torch.abs(
            adv_patch[:, 1:, :] - adv_patch[:, :-1, :] + 0.000001), 0)
        tvcomp2 = torch.sum(torch.sum(tvcomp2, 0), 0)
        tv = tvcomp1 + tvcomp2
        return tv / torch.numel(adv_patch)


class PatchTransformer(nn.Module):
    """
    这里用于训练，训练的时候补丁可以随机放置
    
    用于对补丁进行各种变换
    本类在测试、训练中使用方式不一样
    PatchTransformer: transforms batch of patches

    Module providing the functionality necessary to transform a batch of patches, randomly adjusting brightness and
    contrast, adding random amount of noise, and rotating randomly. Resizes patches according to as size based on the
    batch of labels, and pads them to the dimension of an image.
    根据labels中的数据对补丁进行缩放，并将其填充到图像上

    """

    def __init__(self):
        super(PatchTransformer, self).__init__()
        '''
        self.min_contrast = 1.  # contrast
        self.max_contrast = 1.
        # 光照
        self.min_brightness = 0.
        self.max_brightness = 0.
        # 随机噪声
        self.noise_factor = 0.0
        # 角度

        self.minangle = -0 / 180 * math.pi
        self.maxangle = 0 / 180 * math.pi  # 固定角度
        '''
        self.min_contrast = 0.8
        self.max_contrast = 1.2
        self.min_brightness = -0.1
        self.max_brightness = 0.1
        self.noise_factor = 0.10
        self.minangle = -180 / 180 * math.pi  # self.minangle = -20 / 180 * math.pi
        self.maxangle = 180 / 180 * math.pi
        self.medianpooler = MedianPool2d(7, same=True)

    def lab_transform(self, lab_batch_origin):
        '''
        对原始的lab_batch进行变换，只保留标签数据中size最大的目标
        input: lab_batch_origin
        output: lab_batch_select 
        '''
        
        
        # 以下的标签是nx5，是经过0.4筛选得到
        lab_batch_select = torch.cuda.FloatTensor(
            lab_batch_origin.size(0), 1, 5).fill_(0)
        # shape = [batch, 1, 5]
        area_cal = lab_batch_origin[:, :, 3] * lab_batch_origin[:, :, 4]
        #   测试数据需要根据是5列还是6列对计算进行区分
        max_value, max_index = torch.max(area_cal, 1)  # 返回每张图片label面积的最大值及索引
        #   如果max_value = 1，则场景中没目标。
        min_value, min_index = torch.min(area_cal, 1)  # 找到面积最小的索引

        for i in range(lab_batch_origin.size(0)):
            if max_value[i] > 0.99:
                lab_batch_select[i, :, :] = torch.tensor(
                    [0.25, 0.25, 0.25, 0.25, 0.25]).cuda()
            else:
                temp_max = lab_batch_origin[i, max_index[i], :]
                temp_min = lab_batch_origin[i, min_index[i], :]
                lab_batch_select[i, :, :] = (temp_max + temp_min) / 2.
                # lab_batch_select[i, :, :] = lab_batch_origin[i, max_index[i], :]
        
        '''
        # 以下标签基于nx7得到，即经过0.01筛选得到的数据。
        
        lab_batch_select = torch.cuda.FloatTensor(
            lab_batch_origin.size(0), 1, 7).fill_(0)
        # shape = [batch, 1, 5]
        area_cal = lab_batch_origin[:, :, 2] * lab_batch_origin[:, :, 3]
        max_value, max_index = torch.max(area_cal, 1)  # 返回每张图片label面积的最大值及索引
        #   如果max_value = 1，则场景中没目标。#    因为之前的是填充为了1
        min_value, min_index = torch.min(area_cal, 1)  # 找到面积最小的索引

        ##################################################################################
        #   下面是对标签进行筛选和判断，如果场景中无目标，
        ##################################################################################
        if len(lab_batch_origin[0]) == 1:
            lab_batch_select[0, :, :] = torch.tensor(
                    [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25]).cuda()
        else:
            for i in range(lab_batch_origin.size(0)):
                if max_value[i] > 0.99:
                    lab_batch_select[i, :, :] = torch.tensor(
                        [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25]).cuda()
                else:
                    temp_max = lab_batch_origin[i, max_index[i], :]
                    temp_min = lab_batch_origin[i, min_index[i], :]
                    lab_batch_select[i, :, :] = (temp_max + temp_min) / 2.
                    # lab_batch_select[i, :, :] = lab_batch_origin[i, max_index[i], :]
        '''
        return lab_batch_select
        # return lab_batch_select

    def forward(
            self,
            adv_patch,
            lab_batch,
            img_size,
            do_rotate=True,
            rand_loc=False):
        """
        inputs：3-channels adv_patch, 3维标签（已扩维），图片大小
        lab_batch = [batch, max_lab, 5]

        outputs：[1, max_lab, 3, 608, 608]

        # 在训练和测试时传进来的参数是不一样的
        在训练时，lab_batch=(batch, max_lab, 5)，测试时因为是单张图片进行，
        因此在传入时已经进行了扩维，lab_batch=(1, len(labels), 5)，
        即此时传进来的第一维肯定是1（unsqueeze），第二维和labels数据有几行相关
        """

        adv_patch = self.medianpooler(
            adv_patch.unsqueeze(0))  # 返回[1,3,224,224], 补丁size()是固定的

        pad = (img_size - adv_patch.size(-1)) / 2  # (608-224) / 2 = 192
        # Make a batch of patches
        adv_patch = adv_patch.unsqueeze(0)  # .unsqueeze(0)  5维，[1,1,3,224,224]
        adv_batch = adv_patch.expand(
            lab_batch.size(0), 1, -1, -1, -1)
        # print("adv_patch size : ", adv_batch.size())
        # adv_batch = adv_patch.expand(
        #     lab_batch.size(0), lab_batch.size(1), -1, -1, -1)     #   假设只放置一个补丁，那么expand()的第二个维度就是1
        #   即adv_patch.expand()中第二个参数和具体的问题相关。
        #   [batch, max_lab=1, 3, 224, 224]
        batch_size = torch.Size(
            (lab_batch.size(0), 1))  # [batch, max_lab]
        #   调整成[batch, 1]

        contrast = torch.cuda.FloatTensor(batch_size).uniform_(
            self.min_contrast, self.max_contrast)
        #   [batch, max_lab]
        # [batch, max_lab, 1, 1, 1]
        contrast = contrast.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        #   这里是在(-1)维扩维

        contrast = contrast.expand(-1, -1, adv_batch.size(-3),
                                   adv_batch.size(-2), adv_batch.size(-1))
        contrast = contrast.cuda()  # [batch, 1, 3, 224, 224]
        #   max_lab = 1

        # Create random brightness tensor
        brightness = torch.cuda.FloatTensor(batch_size).uniform_(
            self.min_brightness, self.max_brightness)
        brightness = brightness.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        brightness = brightness.expand(-1, -1, adv_batch.size(-3),
                                       adv_batch.size(-2), adv_batch.size(-1))
        brightness = brightness.cuda()  # [1,2,3,300,300]
        noise = torch.cuda.FloatTensor(
            adv_batch.size()).uniform_(-1, 1) * self.noise_factor  # [1,2,3,300,300]
        # 更改代码，添加随机均匀分布噪声
        # Apply contrast/brightness/noise, clamp
        adv_batch = adv_batch * contrast + brightness + noise  # real_test时固定

        # [1,2,3,300,300]  #  [0.000001, 0.99999]
        adv_batch = torch.clamp(adv_batch, 0.0, 1.)
        '''
        cls_ids = torch.narrow(lab_batch, 2, 0, 1)    #    [batch, max_lab, 1]
        #   torch.narrow()是什么作用？, lab_batch = [batch, max_lab, 5]
        #   torch.narrow(inputs, dim, start, length)，对input进行提取（切片处理）
        #   如上所示，lab_batch=[batch,max_lab,5]，最第三维进行提取，提取第0~1(也就是第一列)数据

        cls_mask = cls_ids.expand(-1, -1, 3)    #   [batch, max_lab, 3]
        cls_mask = cls_mask.unsqueeze(-1)  # [batch, max_lab, 3, 1], 最后一个维度扩维
        cls_mask = cls_mask.expand(-1, -1, -1,
                                   adv_batch.size(3))  # [batch, max_lab, 3, 224]
        cls_mask = cls_mask.unsqueeze(-1)

        cls_mask = cls_mask.expand(-1, -1, -1, -1,
                                   adv_batch.size(4))  # [batch,max_lab,3,224, 224]
        #   
        mask_size = cls_mask.size()
        print("mask size : ", mask_size)
        msk_batch_test = torch.cuda.FloatTensor(cls_mask.size()).fill_(1)
        # print("size of msk_batch_test :", msk_batch_test.size())  # debug
        msk_batch = torch.cuda.FloatTensor(cls_mask.size()).fill_(1)
        
        #   假设只是要得到和adv_patch同维度全1元素的tensor，大可不必这么费劲？？
        '''
        msk_batch = torch.ones_like(adv_batch).cuda()
        # print('mask_batch siz : ', msk_batch.size())

        mypad = nn.ConstantPad2d(
            (int(pad + 0.5), int(pad), int(pad + 0.5), int(pad)), 0)

        adv_batch = mypad(adv_batch)
        msk_batch = mypad(msk_batch)  # [batch, max_lab, 3, 608, 608]
        # anglesize = (lab_batch.size(0) * lab_batch.size(1))  #
        anglesize = lab_batch.size(0)
        # anglesize_ = lab_batch.size()[0]
        if do_rotate:
            angle = torch.cuda.FloatTensor(anglesize).uniform_(
                self.minangle, self.maxangle)  # 2
            # 将tensor用从均匀分布中的值填充
        else:
            angle = torch.cuda.FloatTensor(anglesize).fill_(0)

        # Resizes and rotates，调整大小并缩放
        current_patch_size = adv_patch.size(-1)  # 224
        # lab_batch_scaled = torch.cuda.FloatTensor(
        #     lab_batch.size()).fill_(0)  # [batch, max_lab, 5]   #   全 0 tensor
        #   这里学习torch.cuda.FloatTensor()语法
        lab_batch_scaled = torch.cuda.FloatTensor(lab_batch.size(
            0), 1, 5).fill_(0)  # [batch, max_lab, 5]   #   全0tensor

        #   如果场景中没目标，即lab_batch_scaled = [1,1,1,1,1]
        #   以下要对lab_batch进行筛选，只保留最大size的lab
        lab_batch_selected = self.lab_transform(lab_batch)  # 调用函数对其进行变换

        '''
        #   以下进行还原，用于计算缩放系数
        lab_batch_scaled[:, :, 1] = lab_batch_selected[:,
                                                       :, 1] * img_size  # [batch,1,5]
        lab_batch_scaled[:, :, 2] = lab_batch_selected[:,
                                                       :, 2] * img_size  # 第2、3（1，2）列没用上

        lab_batch_scaled[:, :, 3] = lab_batch_selected[:, :, 3] * img_size
        lab_batch_scaled[:, :, 4] = lab_batch_selected[:, :, 4] * img_size

        #---------------------------------------------------------------#
        #   定义
        #---------------------------------------------------------------#
        pre_scale = SCALE_FACTOR  # 缩放因子
        #   获得缩放因子
        print("pre_scale : ", pre_scale)
        target_size = torch.sqrt(((lab_batch_scaled[:, :, 3].mul(
            1 / pre_scale)) ** 2) + ((lab_batch_scaled[:, :, 4].mul(1 / pre_scale)) ** 2))  # [batch, 1]

        '''
        '''   
        这里需要注意，因为测试和训练的时候数据格式不一样，
        训练标签: [id, x, y, w, h]
        测试标签：[x, y, w, h, obj, cls, id]
        '''
        
        lab_batch_scaled[:, :, 0] = lab_batch_selected[:,
                                                       :, 0] * img_size  # [batch,1,5]
        lab_batch_scaled[:, :, 1] = lab_batch_selected[:,
                                                       :, 1] * img_size  # 第2、3（1，2）列没用上

        lab_batch_scaled[:, :, 2] = lab_batch_selected[:, :, 2] * img_size
        lab_batch_scaled[:, :, 3] = lab_batch_selected[:, :, 3] * img_size

        #---------------------------------------------------------------#
        #   定义
        #---------------------------------------------------------------#
        pre_scale = SCALE_FACTOR  # 缩放因子
        #   获得缩放因子
        target_size = torch.sqrt(((lab_batch_scaled[:, :, 2].mul(
            1 / pre_scale)) ** 2) + ((lab_batch_scaled[:, :, 3].mul(1 / pre_scale)) ** 2))  # [batch, 1]
        
        '''
        #   取标签的数据，归一化数据
        target_x = lab_batch_selected[:, :, 1].view(np.prod(batch_size))  # [batch, 1]
        # print(target_x.size(0))
        target_y = lab_batch_selected[:, :, 2].view(np.prod(batch_size))
        #   np.prod()，对括号内元素进行内积操作
        targetoff_x = lab_batch_selected[:, :, 3].view(np.prod(batch_size))  #   w
        targetoff_y = lab_batch_selected[:, :, 4].view(np.prod(batch_size))  #   h
        #   
        
        off_x = targetoff_x / 2.    #   得到边长的一半
        target_x_right = target_x + off_x  
        target_x_left = target_x - off_x

        off_y = targetoff_y / 2.
        target_y_lower = target_y + off_y
        target_y_upper = target_y - off_y
        '''
        #####################################################################################
        #   隔断
        #####################################################################################

        #   这里的位置现在是随机
        target_x = torch.rand(lab_batch_scaled[:, :, 1].size()).cuda()
        target_x = target_x.view(np.prod(batch_size))
        #   这里的rand(size=)，对于x,y而言，size是一样的
        # target_x = torch.rand_like(lab_batch_scaled[:,:,1])
        #   位置随机的话，超出边界如何处理？
        #   认为设置边界，当超出边界时，直接取边界，边界设置为[0.2, 0.8]
        target_y = torch.rand(lab_batch_scaled[:, :, 2].size()).cuda()

        target_y = target_y.view(np.prod(batch_size))

        target_x = torch.max(target_x, torch.tensor(0.2).cuda())
        #   torch.max/min除了获得tensor中的最大、最小值外，还可以对值进行比较，从而返回最大或最小值
        #   例如torch.max(a, b)返回相应元素的最大值，即谁大保留谁
        target_y = torch.min(target_y, torch.tensor(0.8).cuda())
        #   通过上面的操作可将中心坐标约束在[0.2,0.8]内

        #####################################################################################
        #   这里的target_x, target_y就是现在补丁的归一化坐标了吧
        #####################################################################################
        patch_center_x = (target_x * img_size).view(-1, 1)
        patch_center_y = (target_y * img_size).view(-1, 1)

        patch_center_axis = torch.cat([patch_center_x, patch_center_y], 1)

        scale = target_size / current_patch_size  # 最终的缩放因子
        scale = scale.view(anglesize)

        s = adv_batch.size()  # [batch,1,3,608,608]
        adv_batch = adv_batch.view(
            s[0] * s[1], s[2], s[3], s[4])  # [batch*max_lab,3,608,608]
        msk_batch = msk_batch.view(
            s[0] * s[1], s[2], s[3], s[4])  # [batch*max_lab,3,608,608]

        tx = (-target_x + 0.5) * 2  # 两个数  # 这个2和theta/affine_grid机制相关
        ty = (-target_y + 0.5) * 2  # 和仿射变换的坐标相关，左上角为原点(0.,0.)
        #   向右、向下为负，这里的系数2和affine()函数定义相关
        #   creation attack时，位置要随机，就和lab中的位置信息不再相关

        sin = torch.sin(angle)
        cos = torch.cos(angle)

        theta = torch.cuda.FloatTensor(anglesize, 2, 3).fill_(0)
        
        
        # theta = torch.zeros(anglesize, 2, 3)  # simpler # theta默认是一个2x3的矩阵
        theta[:, 0, 0] = cos / scale
        theta[:, 0, 1] = sin / scale
        theta[:, 0, 2] = tx * cos / scale + ty * sin / scale
        theta[:, 1, 0] = -sin / scale
        theta[:, 1, 1] = cos / scale
        theta[:, 1, 2] = -tx * sin / scale + ty * cos / scale
                
        grid = F.affine_grid(theta, adv_batch.shape)  # 第二个参数设置图片大小
        # 同时形状不变（第二个参数）
        # grid = F.affine_grid(theta, adv_batch.shape)
        adv_batch_t = F.grid_sample(adv_batch, grid)
        msk_batch_t = F.grid_sample(msk_batch, grid)
        '''
        theta1 = torch.cuda.FloatTensor(anglesize, 2, 3).fill_(0)
        # theta = torch.zeros(anglesize, 2, 3)  # simpler # theta默认是一个2x3的矩阵
        #   这个矩阵是旋转、缩放和平移同时操作，但是为了位置不重叠，
        #   1. 先做缩放和旋转，再做平移
        #   2. 先做缩放，再所旋转和平移
        #   theta1：只做旋转和缩放

        theta1[:, 0, 0] = cos / scale
        theta1[:, 0, 1] = sin / scale
        theta1[:, 0, 2] = 0
        theta1[:, 1, 0] = -sin / scale
        theta1[:, 1, 1] = cos / scale
        theta1[:, 1, 2] = 0

        grid = F.affine_grid(theta1, adv_batch.shape)  # 第二个参数设置图片大小
        # 同时形状不变（第二个参数）
        # grid = F.affine_grid(theta, adv_batch.shape)
        adv_batch_t = F.grid_sample(adv_batch, grid)
        msk_batch_t = F.grid_sample(msk_batch, grid)
        
        theta2 = torch.cuda.FloatTensor(anglesize, 2, 3).fill_(0)
        theta2[:, 0, 0] = 1
        theta2[:, 0, 1] = 0
        theta2[:, 0, 2] = tx  # 平移参数设置
        # theta2[:, 0, 2] = 0
        theta2[:, 1, 0] = 0
        theta2[:, 1, 1] = 1
        theta2[:, 1, 2] = ty
        # theta2[:, 1, 2] = 0
        grid2 = F.affine_grid(theta2, adv_batch.shape)
        adv_batch_t = F.grid_sample(adv_batch_t, grid2)
        msk_batch_t = F.grid_sample(msk_batch_t, grid2)
        '''

        adv_batch_t = adv_batch_t.view(
            s[0], s[1], s[2], s[3], s[4])  # [batch,max_lab,3,608,608]
        msk_batch_t = msk_batch_t.view(
            s[0], s[1], s[2], s[3], s[4])  # [batch,max_lab,3,608,608]

        # [batch,max_lab,3,608,608] # 000001
        adv_batch_t = torch.clamp(adv_batch_t, 0.0, 1.)
        adv_patch_mask = adv_batch_t * msk_batch_t

        return adv_patch_mask, patch_center_axis


class PatchApplier(nn.Module):
    """向图片上添加补丁
    PatchApplier: applies adversarial patches to images.

    Module providing the functionality necessary to apply a patch to all detections in all images in the batch.

    """

    def __init__(self):
        super(PatchApplier, self).__init__()

    def forward(self, img_batch, adv_batch):
        #  将patch应用到图片上

        # 输入[batch, max_lab, 3, 608, 608],

        advs = torch.unbind(adv_batch, 1)
        # 按第二维进行切片，即max_lab，最后advs是一个补丁对那个每个lab的集合
        #  返回指定维度切片后的元组，
        for adv in advs:
            #
            # adv==0时保留img_batch,否则保留adv（补丁处是补丁，其它地方是原图片）
            # print("adv : ", adv)
            img_batch = torch.where((adv == 0.), img_batch, adv)
            '''
            adv=[1,3,608,608]
            说明：首先这里的img_batch不一定要求维度和adv一致，这应该是torch.where函数的问题
            当img_batch也是[1,3,608,608],返回的也是[1,3,608,608]
            当img_batch是[3,608,608]，返回的也是[1,3,608,608]

            '''
            # 对每一个框进行操作，如果没有框，对应的补丁全是0，所以也不影响
            """
            torch.where(condition,a,b)
            合并a,b两个tensor，满足条件下保留a，否则是b（元素替换）
            """
        return img_batch


'''
class PatchGenerator(nn.Module):
    """PatchGenerator: network module that generates adversarial patches.

    Module representing the neural network that will generate adversarial patches.

    """

    def __init__(self, cfgfile, weightfile, img_dir, lab_dir):
        super(PatchGenerator, self).__init__()
        self.yolo = Darknet(cfgfile).load_weights(weightfile)
        self.dataloader = torch.utils.data.DataLoader(InriaDataset(img_dir, lab_dir, shuffle=True),
                                                      batch_size=5,
                                                      shuffle=True)
        self.patchapplier = PatchApplier()
        self.nmscalculator = NMSCalculator()
        self.totalvariation = TotalVariation()

    def forward(self, *input):
        pass
'''


class DotaDataset(Dataset):
    #   对数据集进行调整，仍然是输入max_lab，但是此时会设置max_lab=1，从而保证只在
    """读取数据集
    InriaDataset: representation of the INRIA person dataset.

    Internal representation of the commonly used INRIA person dataset.
    Available at: http://pascal.inrialpes.fr/data/human/

    Attributes:
        len: An integer number of elements in the
        img_dir: Directory containing the images of the INRIA dataset.
        lab_dir: Directory containing the labels of the INRIA dataset.
        img_names: List of all image file names in img_dir.
        shuffle: Whether or not to shuffle the dataset.

    """

    def __init__(self, img_dir, lab_dir, max_lab, imgsize, shuffle=True):
        """
        输入参数包括：图片地址、标签地址，最大标签，图片尺寸
        """
        n_png_images = len(fnmatch.filter(os.listdir(img_dir), '*.png'))
        n_jpg_images = len(fnmatch.filter(os.listdir(img_dir), '*.jpg'))
        n_images = n_png_images + n_jpg_images
        n_labels = len(fnmatch.filter(os.listdir(lab_dir), '*.txt'))

        print("n_images = ", n_images, '\n', "n_labels = ", n_labels)  # test

        assert n_images == n_labels, "Number of images and number of labels does't match"
        # 这个地方会做出判断，如果图片数量和标签数不一样则报错
        self.len = n_images
        self.img_dir = img_dir
        self.lab_dir = lab_dir
        self.imgsize = imgsize
        self.img_names = fnmatch.filter(os.listdir(
            img_dir), '*.png') + fnmatch.filter(os.listdir(img_dir), '*.jpg')
        self.shuffle = shuffle
        self.img_paths = []  # 填入路径
        for img_name in self.img_names:
            self.img_paths.append(os.path.join(self.img_dir, img_name))
        self.lab_paths = []
        for img_name in self.img_names:
            lab_path = os.path.join(self.lab_dir, img_name).replace(
                '.jpg', '.txt').replace('.png', '.txt')
            self.lab_paths.append(lab_path)
        # 分别得到image和lab的路径和名称
        self.max_n_labels = max_lab  # 最大标签数

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        assert idx <= len(self), 'index range error'
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        lab_path = os.path.join(self.lab_dir, self.img_names[idx]).replace(
            '.jpg', '.txt').replace('.png', '.txt')
        # 分别得到图片和lab的地址
        image = Image.open(img_path).convert('RGB')
        # check to see if label file contains data.
        if os.path.getsize(lab_path):
            label = np.loadtxt(lab_path)
        # 若lab为空，直接注释掉行不行？
        else:
            label = np.ones([5])
            # label = np.ones([0])

        label = torch.from_numpy(label).float()
        if label.dim() == 1:
            label = label.unsqueeze(0)

        image, label = self.pad_and_scale(image, label)
        # 这个地方对图片进行预处理，所以训练的时候直接使用处理过后的图片就行
        transform = transforms.ToTensor()  # 这个地方转换为tensor，
        image = transform(image)
        # 需要对image和lab进行pad
        label = self.pad_lab(label)
        return image, label

    def pad_and_scale(self, img, lab):
        """

        Args:
            img:

        Returns:

        """
        w, h = img.size
        if w == h:
            padded_img = img  #
        else:
            dim_to_pad = 1 if w < h else 2
            if dim_to_pad == 1:
                padding = (h - w) / 2
                padded_img = Image.new('RGB', (h, h), color=(127, 127, 127))
                padded_img.paste(img, (int(padding), 0))
                lab[:, [1]] = (lab[:, [1]] * w + padding) / h
                lab[:, [3]] = (lab[:, [3]] * w / h)
            else:
                padding = (w - h) / 2
                padded_img = Image.new('RGB', (w, w), color=(127, 127, 127))
                padded_img.paste(img, (0, int(padding)))
                lab[:, [2]] = (lab[:, [2]] * h + padding) / w
                lab[:, [4]] = (lab[:, [4]] * h / w)
        # 根据输入图片对原始图片进行resize和padding
        resize = transforms.Resize((self.imgsize, self.imgsize))
        padded_img = resize(padded_img)  # choose here
        return padded_img, lab

    def pad_lab(self, lab):
        pad_size = self.max_n_labels - lab.shape[0]  # 最终需要填充的labels数据
        if(pad_size > 0):
            # padded_lab = F.pad(lab, (0, 0, 0, pad_size),
            #                    value=1)  # 给labels中的其他数据填充为1
            # 下面行填充为1
            padded_lab = F.pad(lab, (0, 0, 0, pad_size),
                               value=1e-6)  # 给labels中的其他数据填充为1
        else:
            padded_lab = lab
        return padded_lab
        '''
        torch.nn.functional.pad(input, pad, mode, value)
        pad：表示填充方式，分别表示左、右、上、下，此时是下填充
        '''


class PatchTransformer_vanishing(nn.Module):
    """用于对补丁进行各种变换
    本类在测试、训练中使用方式不一样
    PatchTransformer: transforms batch of patches

    Module providing the functionality necessary to transform a batch of patches, randomly adjusting brightness and
    contrast, adding random amount of noise, and rotating randomly. Resizes patches according to as size based on the
    batch of labels, and pads them to the dimension of an image.
    根据labels中的数据对补丁进行缩放，并将其填充到图像上

    """

    def __init__(self):
        super(PatchTransformer_vanishing, self).__init__()
        '''
        self.min_contrast = 1.  # contrast
        self.max_contrast = 1.
        # 光照
        self.min_brightness = 0.
        self.max_brightness = 0.
        # 随机噪声
        self.noise_factor = 0.0
        # 角度

        self.minangle = -0 / 180 * math.pi
        self.maxangle = 0 / 180 * math.pi  # 固定角度
        '''
        self.min_contrast = 0.8
        self.max_contrast = 1.2
        self.min_brightness = -0.1
        self.max_brightness = 0.1
        self.noise_factor = 0.10
        self.minangle = -180 / 180 * math.pi  # self.minangle = -20 / 180 * math.pi
        self.maxangle = 180 / 180 * math.pi
        self.medianpooler = MedianPool2d(7, same=True)

    def forward(
            self,
            adv_patch,
            lab_batch,
            img_size,
            do_rotate=True,
            rand_loc=False,
            orient=None,
            test_real=False):
        """
        inputs：3_channels adv_patch, 3维标签（已扩维），图片大小
                lab_batch = [batch, max_lab, 5]
        outputs：[1, max_lab, 3, 608, 608]

        # 在训练和测试时传进来的参数是不一样的
        在训练时，lab_batch=(batch, max_lab, 5)，测试时因为是单张图片进行，
        因此在传入时已经进行了扩维，lab_batch=(1, len(labels), 5)，
        即此时传进来的第一维肯定是1（unsqueeze），第二维和labels数据有几行相关
        """

        adv_patch = self.medianpooler(
            adv_patch.unsqueeze(0))  # 返回[1,3,300,300], 补丁size()是固定的

        pad = (img_size - adv_patch.size(-1)) / 2  # (608-300) / 2 = 154
        # Make a batch of patches
        adv_patch = adv_patch.unsqueeze(0)  # .unsqueeze(0)  5维，[1,1,3,300,300]
        adv_batch = adv_patch.expand(
            lab_batch.size(0), lab_batch.size(1), -1, -1, -1)
        batch_size = torch.Size(
            (lab_batch.size(0), lab_batch.size(1)))  # [batch, max_lab]

        contrast = torch.cuda.FloatTensor(batch_size).uniform_(
            self.min_contrast, self.max_contrast)
        contrast = contrast.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        contrast = contrast.expand(-1, -1, adv_batch.size(-3),
                                   adv_batch.size(-2), adv_batch.size(-1))
        contrast = contrast.cuda()  # [1,2,3,300,300]

        # Create random brightness tensor
        brightness = torch.cuda.FloatTensor(batch_size).uniform_(
            self.min_brightness, self.max_brightness)
        brightness = brightness.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        brightness = brightness.expand(-1, -1, adv_batch.size(-3),
                                       adv_batch.size(-2), adv_batch.size(-1))
        brightness = brightness.cuda()  # [1,2,3,300,300]
        noise = torch.cuda.FloatTensor(
            adv_batch.size()).uniform_(-1, 1) * self.noise_factor  # [1,2,3,300,300]
        # 更改代码，添加随机均匀分布噪声
        # Apply contrast/brightness/noise, clamp
        if test_real:  # 测试和训练不同
            adv_batch = adv_batch
        else:
            adv_batch = adv_batch * contrast + brightness + noise  # real_test时固定

        # [1,2,3,300,300]  #  [0.000001, 0.99999]
        adv_batch = torch.clamp(adv_batch, 0.0, 1.)
        cls_ids = torch.narrow(lab_batch, 2, 0, 1)

        cls_mask = cls_ids.expand(-1, -1, 3)
        cls_mask = cls_mask.unsqueeze(-1)  # [1,2,3,1], 最后一个维度扩维
        cls_mask = cls_mask.expand(-1, -1, -1,
                                   adv_batch.size(3))  # [1,2,3,300]
        cls_mask = cls_mask.unsqueeze(-1)

        cls_mask = cls_mask.expand(-1, -1, -1, -1,
                                   adv_batch.size(4))  # [1,2,3,300,300]
        msk_batch_test = torch.cuda.FloatTensor(cls_mask.size()).fill_(1)
        # print("size of msk_batch_test :", msk_batch_test.size())  # debug
        msk_batch = torch.cuda.FloatTensor(cls_mask.size()).fill_(1)
        mypad = nn.ConstantPad2d(
            (int(pad + 0.5), int(pad), int(pad + 0.5), int(pad)), 0)

        adv_batch = mypad(adv_batch)
        msk_batch = mypad(msk_batch)  # [1,2,3,608,608]
        anglesize = (lab_batch.size(0) * lab_batch.size(1))  # 2，旋转角度数量和预测框个数一致
        if do_rotate:
            angle = torch.cuda.FloatTensor(anglesize).uniform_(
                self.minangle, self.maxangle)  # 2
            # 将tensor用从均匀分布中的值填充
        else:
            angle = torch.cuda.FloatTensor(anglesize).fill_(0)

        # Resizes and rotates，调整大小并缩放
        current_patch_size = adv_patch.size(-1)  # 300
        lab_batch_scaled = torch.cuda.FloatTensor(
            lab_batch.size()).fill_(0)  # [1,2,5]全0tensor
        lab_batch_scaled[:, :, 1] = lab_batch[:, :, 1] * img_size  # [1,2,5]
        lab_batch_scaled[:, :, 2] = lab_batch[:, :, 2] * img_size  # 第2、3列没用上

        lab_batch_scaled[:, :, 3] = lab_batch[:, :, 3] * img_size
        lab_batch_scaled[:, :, 4] = lab_batch[:, :, 4] * img_size

        #---------------------------------------------------------------#
        #   定义
        #---------------------------------------------------------------#
        pre_scale = 8.0  # 缩放因子
        # pre_scale = 8.0 * 1.414 #   粘贴两个补丁，同时使得面积保持一致大小
        # 要使得面积是1/4，pre_scale是1/2即可
        target_size = torch.sqrt(((lab_batch_scaled[:, :, 3].mul(
            1 / pre_scale)) ** 2) + ((lab_batch_scaled[:, :, 4].mul(1 / pre_scale)) ** 2))  # [1,2]

        target_x = lab_batch[:, :, 1].view(np.prod(batch_size))  # [1, 2]
        target_y = lab_batch[:, :, 2].view(np.prod(batch_size))

        targetoff_x = lab_batch[:, :, 3].view(np.prod(batch_size))  # w
        targetoff_y = lab_batch[:, :, 4].view(np.prod(batch_size))  # h
        # print("size of target_x :", target_x.size())  # debug
        '''  这部分代码可注释
        '''

        if(rand_loc):  # 位置随机
            off_x = targetoff_x * \
                (torch.cuda.FloatTensor(targetoff_x.size()
                                        ).uniform_(-0.2, 0.2))  # 后面的随机太大
            target_x = target_x + off_x  # 加了随机性
            # if target_x < bound_x1:
            #     target_x = bound_x1
            # if target_x > bound_x2:
            #     target_x = bound_x2
            off_y = targetoff_y * \
                (torch.cuda.FloatTensor(targetoff_y.size()).uniform_(-0.2, 0.2))
            # 位置比较关键，想让其偏下，试试参数
            target_y = target_y + off_y
            # if target_y < bound_y1:
            #     target_y = bound_y1
            # if target_y > bound_y2:
            #     target_y = bound_y2

        scale = target_size / current_patch_size  # 最终的缩放因子
        scale = scale.view(anglesize)

        s = adv_batch.size()  # [1,2,3,608,608]
        adv_batch = adv_batch.view(
            s[0] * s[1], s[2], s[3], s[4])  # [batch*max_lab,3,608,608]
        msk_batch = msk_batch.view(
            s[0] * s[1], s[2], s[3], s[4])  # [batch*max_lab,3,608,608]

        if orient == "left":  # 左边
            target_x = target_x - targetoff_x / 6.0  # target_x是中心点坐标
            # 水平4等分或6等分排列
        elif orient == "right":  # 右边
            target_x = target_x + targetoff_x / 6.0

        tx = (-target_x + 0.5) * 2  # 两个数  # 这个2和theta/affine_grid机制相关
        ty = (-target_y + 0.5) * 2

        sin = torch.sin(angle)
        cos = torch.cos(angle)

        #   单阶段实现
        theta = torch.cuda.FloatTensor(anglesize, 2, 3).fill_(0)
        # theta = torch.zeros(anglesize, 2, 3)  # simpler # theta默认是一个2x3的矩阵
        theta[:, 0, 0] = cos / scale
        theta[:, 0, 1] = sin / scale
        theta[:, 0, 2] = tx * cos / scale + ty * sin / scale
        theta[:, 1, 0] = -sin / scale
        theta[:, 1, 1] = cos / scale
        theta[:, 1, 2] = -tx * sin / scale + ty * cos / scale

        grid = F.affine_grid(theta, adv_batch.shape)  # 第二个参数设置图片大小
        # 同时形状不变（第二个参数）
        # grid = F.affine_grid(theta, adv_batch.shape)
        adv_batch_t = F.grid_sample(adv_batch, grid)
        msk_batch_t = F.grid_sample(msk_batch, grid)

        #   两阶段实现
        # theta1 = torch.cuda.FloatTensor(anglesize, 2, 3).fill_(0)
        # # theta = torch.zeros(anglesize, 2, 3)  # simpler # theta默认是一个2x3的矩阵
        # #   这个矩阵是旋转、缩放和平移同时操作，但是为了位置不重叠，
        # #   1. 先做缩放和旋转，再做平移
        # #   2. 先做缩放，再所旋转和平移
        # #   theta1：只做旋转和缩放

        # theta1[:, 0, 0] = cos / scale
        # theta1[:, 0, 1] = sin / scale
        # theta1[:, 0, 2] = 0
        # theta1[:, 1, 0] = -sin / scale
        # theta1[:, 1, 1] = cos / scale
        # theta1[:, 1, 2] = 0

        # grid = F.affine_grid(theta1, adv_batch.shape)  # 第二个参数设置图片大小
        # # 同时形状不变（第二个参数）
        # # grid = F.affine_grid(theta, adv_batch.shape)
        # adv_batch_t = F.grid_sample(adv_batch, grid)
        # msk_batch_t = F.grid_sample(msk_batch, grid)

        # #   定义theta2，用于平移
        # theta2 = torch.cuda.FloatTensor(anglesize, 2, 3).fill_(0)
        # theta2[:, 0, 0] = 1
        # theta2[:, 0, 1] = 0
        # theta2[:, 0, 2] = tx  # 平移参数设置
        # # theta2[:, 0, 2] = 0
        # theta2[:, 1, 0] = 0
        # theta2[:, 1, 1] = 1
        # theta2[:, 1, 2] = ty
        # # theta2[:, 1, 2] = 0
        # grid2 = F.affine_grid(theta2, adv_batch.shape)
        # adv_batch_t = F.grid_sample(adv_batch_t, grid2)
        # msk_batch_t = F.grid_sample(msk_batch_t, grid2)  # 上下两行需要成对出现

        adv_batch_t = adv_batch_t.view(
            s[0], s[1], s[2], s[3], s[4])  # [batch,max_lab,3,608,608]
        msk_batch_t = msk_batch_t.view(
            s[0], s[1], s[2], s[3], s[4])  # [batch,max_lab,3,608,608]

        # [batch,max_lab,3,608,608] # 000001
        adv_batch_t = torch.clamp(adv_batch_t, 0.0, 1.)
        temp_val = adv_batch_t * msk_batch_t

        return adv_batch_t * msk_batch_t


class PatchTransformer_test_mode(nn.Module):
    """
    测试时候用到
    - 对补丁进行变换；
    - 筛选参考检测框；
    - 粘贴补丁，**但是要避免和已有检测框重叠**
    根据labels中的数据对补丁进行缩放，并将其填充到图像上

    """

    def __init__(self, test_mode=False):
        super(PatchTransformer_test_mode, self).__init__()

        self.min_contrast = 0.8
        self.max_contrast = 1.2
        self.min_brightness = -0.1
        self.max_brightness = 0.1
        self.noise_factor = 0.10
        self.minangle = -180 / 180 * math.pi  # self.minangle = -20 / 180 * math.pi
        self.maxangle = 180 / 180 * math.pi

        if test_mode == True:
            '''
            这里是用于控制旋转角度，避免过大（可能）影响效果
            '''
            self.maxangle = 90 / 180 * math.pi
            self.minangle = -90 / 180 * math.pi
        self.medianpooler = MedianPool2d(7, same=True)

    def lab_transform(self, lab_batch_origin):
        '''
        对原始的lab_batch进行变换，只保留标签数据中size最大的目标
        input: lab_batch_origin
        output: lab_batch_select 
        '''
        
        '''
        以下的标签是nx5，是经过0.4筛选得到，用于训练
        lab_batch_select = torch.cuda.FloatTensor(
            lab_batch_origin.size(0), 1, 5).fill_(0)
        # shape = [batch, 1, 5]
        area_cal = lab_batch_origin[:, :, 3] * lab_batch_origin[:, :, 4]
        max_value, max_index = torch.max(area_cal, 1)  # 返回每张图片label面积的最大值及索引
        #   如果max_value = 1，则场景中没目标。
        min_value, min_index = torch.min(area_cal, 1)  # 找到面积最小的索引

        ##################################################################################
        #   下面是对标签进行筛选和判断，如果场景中无目标，
        ##################################################################################
        for i in range(lab_batch_origin.size(0)):
            if max_value[i] > 0.99:
                lab_batch_select[i, :, :] = torch.tensor(
                    [0.25, 0.25, 0.25, 0.25, 0.25]).cuda()
            else:
                temp_max = lab_batch_origin[i, max_index[i], :]
                temp_min = lab_batch_origin[i, min_index[i], :]
                lab_batch_select[i, :, :] = (temp_max + temp_min) / 2.
                # lab_batch_select[i, :, :] = lab_batch_origin[i, max_index[i], :]
        '''
        '''
        以下标签基于nx7得到，即经过0.01筛选得到的数据。用于测试
        '''
        lab_batch_select = torch.cuda.FloatTensor(
            lab_batch_origin.size(0), 1, 7).fill_(0)
        # shape = [batch, 1, 5]
        area_cal = lab_batch_origin[:, :, 2] * lab_batch_origin[:, :, 3]
        max_value, max_index = torch.max(area_cal, 1)  # 返回每张图片label面积的最大值及索引
        #   如果max_value = 1，则场景中没目标。
        min_value, min_index = torch.min(area_cal, 1)  # 找到面积最小的索引

        ##################################################################################
        #   下面是对标签进行筛选和判断，如果场景中无目标，
        ##################################################################################
        if len(lab_batch_origin[0]) == 1:
            lab_batch_select[0, :, :] = torch.tensor(
                    [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25]).cuda()
        else:
            for i in range(lab_batch_origin.size(0)):
                if max_value[i] > 0.99:
                    lab_batch_select[i, :, :] = torch.tensor(
                        [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25]).cuda()
                else:
                    temp_max = lab_batch_origin[i, max_index[i], :]
                    temp_min = lab_batch_origin[i, min_index[i], :]
                    lab_batch_select[i, :, :] = (temp_max + temp_min) / 2.
                    # lab_batch_select[i, :, :] = lab_batch_origin[i, max_index[i], :]
                
        return lab_batch_select

    def inter_axis_cal(self, lab_batch, semi_edge, img_size):
        '''
        根据所有的label和现在patch的大小、位置、旋转等
        返回干涉位置
        input: 
        lab_batch: 图片中所有标签数据，1 x len x 7，因为在粘贴前进行了unsqueeze，归一化数据
        [x, y, w, h, obj_conf, cls_conf, id]
        semi_edge: 补丁外接正方形的半边长
        1. 补丁与周边不重叠
        2. 补丁与所有lab不重叠
        这里和具体的补丁无关
        '''
        lab_batch = lab_batch.squeeze(0)  # 预期是[lab_len, 5]
        #   将之前升维的压缩回去
        lab_scale = lab_batch * img_size  # 还原到图片空间
        #   计算面积
        # lab_area = lab_scale[:, 3] * lab_scale[:, 4]  #   原来0.4标签
        ##############################################
        lab_area = lab_scale[:, 2] * lab_scale[:, 3]    #   0.01标签
        #   根据面积进行排序
        _, sorted_index = torch.sort(lab_area)  
        #   默认为升序，正好和现在的需求一致，先从小的开始干涉计算
        #   分别返回排序后的值和索引，目前只需要索引即可，具体值不需要
        # print("after sorted : ", sorted_value, "and index : ", sorted_index)
        len_lab = len(lab_batch)    #   这里也必须得先squeeze，才能得到正确的维度
        
        #   生成一个和len_lab同维的模板矩阵，然后每一维度进行干涉计算
        #   这里和之前的补丁粘贴思想有点类似，分块处理
        temp_lab = torch.zeros([len_lab, img_size, img_size])
        #   对于图片，其channel不能大于4，因此需要手动修改通道数为3，如下所示进行示例说明
        # temp_lab = torch.zeros([3, img_size, img_size])
        
        #   首先，对边缘进行填充，对边缘一定范围都填充为1
        #   abnormal
        # temp_lab[:, 0:int(semi_edge), 0:int(semi_edge)] = 1     #   对两个维度的首部进行填充
        # temp_lab[:, 0:int(semi_edge), -int(semi_edge):] = 1 
        #   上下的处理不等价
        # print("int semi edge : ", int(semi_edge))
        #   normal
        temp_lab[:, 0:int(semi_edge), :] = 1     #   对两个维度的首部进行填充
        temp_lab[:, -int(semi_edge):, :] = 1     #   对尾部进行填充
        
        temp_lab[:, :, 0:int(semi_edge)] = 1     #   对两个维度的首部进行填充
        temp_lab[:, :, -int(semi_edge):] = 1 
        
        # tianchong_01 = transforms.ToPILImage()(temp_lab)
        # tianchong_01_name = '_tianchong_05.png'
        # tianchong_01.save(os.path.join("DOTA_creation_test_imgs/patch_location_test", tianchong_01_name))

        # terminate_index = 0
        for i in range(len_lab):
            
            sum_lab = torch.sum(temp_lab, dim=0)
            #   将求和放在最前面判断，这样如果满足了，要保留的是上一次的结果
            zeros_elem_find = torch.nonzero(sum_lab == 0)
            # print("length of zeros_elem_find : ", len(zeros_elem_find))
            if (len(zeros_elem_find) == 0):
                '''
                如果不再存在0元素。
                则for训练需要提前终止。
                然后，返回什么？
                如果当前计算sum后，没有了0元素，则应返回上一次计算得到的sum结果
                '''
                # terminate_index = i
                return torch.sum(temp_lab[0:i-1,:,:],dim=0)     #   有两个return
                # break
            #   对排序后的index进行遍历，
            #   按照面积，从小到大依次填充计算
            # print("sorted i : ", sorted_index[i])
            lab_index = lab_scale[sorted_index[i]]  #   提取其中的排序第i的元素，即反索引到目标
            # print("lab in index : ", lab_index)
            '''
            0.4标签
            lab_center_x = lab_index[1]    #    x
            lab_center_y = lab_index[2]    #    y
            lab_W = lab_index[3]    #   W
            lab_H = lab_index[4]    #   H
            '''
            #   0.01标签数据格式
            lab_center_x = lab_index[0]    #    x
            lab_center_y = lab_index[1]    #    y
            lab_W = lab_index[2]    #   W
            lab_H = lab_index[3]    #   H
            
            temp_lab[i, int(lab_center_x-lab_W/2-semi_edge):int(lab_center_x+lab_W/2+semi_edge),
                     int(lab_center_y-lab_H/2-semi_edge):int(lab_center_y+lab_H/2+semi_edge)] = 1
            #   当只有一个目标的时候，这里会全部填充掉，那么最简单的，其实是在设置缩放补丁那里实现。
            #   减小semi_edge
            #   这里又不能分成下面的两段式
            # temp_lab[i, int(lab_center_x-lab_W/2-semi_edge):int(lab_center_x+lab_W/2+semi_edge),:] = 1
            # temp_lab[i, :, int(lab_center_y-lab_H/2-semi_edge):int(lab_center_y+lab_H/2+semi_edge)] = 1
            
            # tianchong_inner = transforms.ToPILImage()(temp_lab)
            # tianchong_inner_name = '_tianchong_innner_04.png'
            # tianchong_inner.save(os.path.join("DOTA_creation_test_imgs/patch_location_test", tianchong_inner_name))
            
            #   定义在第一维求和变量，用于判断还有没有空间
            #   如果已经全1了，那就说明没有多余的空间用于放置补丁
        # final_zeros_elem_find = torch.nonzero()          
        temp_return = torch.sum(temp_lab, dim=0)
        zeros_elem_find_final = torch.nonzero(temp_return == 0)

        if (len(zeros_elem_find_final) == 0):
            '''
            如果巧了，最终的填充把地方填没了
            那就仍然只保留前len()-1个sum
            '''
            return torch.sum(temp_lab[0:len_lab-1,:,:],dim=0)
        return temp_return

    def forward(
            self,
            adv_patch,
            lab_batch,
            img_size,
            do_rotate=True,
            rand_loc=False):
        """
        inputs：3-channels adv_patch, 3维标签（已扩维），图片大小
        lab_batch = [batch, max_lab, 5]

        outputs：[1, max_lab, 3, 608, 608]

        # 在训练和测试时传进来的参数是不一样的
        在训练时，lab_batch=(batch, max_lab, 5)，测试时因为是单张图片进行，
        因此在传入时已经进行了扩维，lab_batch=(1, len(labels), 5)，
        即此时传进来的第一维肯定是1（unsqueeze），第二维和labels数据有几行相关
        """

        adv_patch = self.medianpooler(
            adv_patch.unsqueeze(0))  # 返回[1,3,224,224], 补丁size()是固定的

        pad = (img_size - adv_patch.size(-1)) / 2  # (608-224) / 2 = 192
        # Make a batch of patches
        adv_patch = adv_patch.unsqueeze(0)  # .unsqueeze(0)  5维，[1,1,3,224,224]
        adv_batch = adv_patch.expand(
            lab_batch.size(0), 1, -1, -1, -1)   #   [lab_batch.size,1,1,3,224,224]

        batch_size = torch.Size(
            (lab_batch.size(0), 1))  # [batch, max_lab]
        #   调整成[batch, 1]

        contrast = torch.cuda.FloatTensor(batch_size).uniform_(
            self.min_contrast, self.max_contrast)
        #   [batch, max_lab]
        # [batch, max_lab, 1, 1, 1]
        contrast = contrast.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        #   这里是在(-1)维扩维

        contrast = contrast.expand(-1, -1, adv_batch.size(-3),
                                   adv_batch.size(-2), adv_batch.size(-1))
        contrast = contrast.cuda()  # [batch, 1, 3, 224, 224]
        #   max_lab = 1

        # Create random brightness tensor
        brightness = torch.cuda.FloatTensor(batch_size).uniform_(
            self.min_brightness, self.max_brightness)
        brightness = brightness.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        brightness = brightness.expand(-1, -1, adv_batch.size(-3),
                                       adv_batch.size(-2), adv_batch.size(-1))
        brightness = brightness.cuda()  # [1,2,3,300,300]
        noise = torch.cuda.FloatTensor(
            adv_batch.size()).uniform_(-1, 1) * self.noise_factor  # [1,2,3,300,300]
        # 更改代码，添加随机均匀分布噪声
        # Apply contrast/brightness/noise, clamp
        # adv_batch = adv_batch * contrast + brightness + noise  # real_test时固定

        # [1,2,3,300,300]  #  [0.000001, 0.99999]
        adv_batch = torch.clamp(adv_batch, 0.0, 1.)     #   [1,1,3,224,224]
        
        #   保存变换后的补丁
        '''
        img_patch_raw = adv_batch[0,0,:,:,:]
        img_patch_raw = transforms.ToPILImage()(img_patch_raw)
        img_patch_raw_savedir = '_patch_raw_01.png'
        img_patch_raw.save(os.path.join("framework/transformer", img_patch_raw_savedir))
        '''
        
        '''
        cls_ids = torch.narrow(lab_batch, 2, 0, 1)    #    [batch, max_lab, 1]
        #   torch.narrow()是什么作用？, lab_batch = [batch, max_lab, 5]
        #   torch.narrow(inputs, dim, start, length)，对input进行提取（切片处理）
        #   如上所示，lab_batch=[batch,max_lab,5]，最第三维进行提取，提取第0~1(也就是第一列)数据

        cls_mask = cls_ids.expand(-1, -1, 3)    #   [batch, max_lab, 3]
        cls_mask = cls_mask.unsqueeze(-1)  # [batch, max_lab, 3, 1], 最后一个维度扩维
        cls_mask = cls_mask.expand(-1, -1, -1,
                                   adv_batch.size(3))  # [batch, max_lab, 3, 224]
        cls_mask = cls_mask.unsqueeze(-1)

        cls_mask = cls_mask.expand(-1, -1, -1, -1,
                                   adv_batch.size(4))  # [batch,max_lab,3,224, 224]
        #   
        mask_size = cls_mask.size()
        print("mask size : ", mask_size)
        msk_batch_test = torch.cuda.FloatTensor(cls_mask.size()).fill_(1)
        # print("size of msk_batch_test :", msk_batch_test.size())  # debug
        msk_batch = torch.cuda.FloatTensor(cls_mask.size()).fill_(1)
        
        #   假设只是要得到和adv_patch同维度全1元素的tensor，大可不必这么费劲？？
        '''
        msk_batch = torch.ones_like(adv_batch).cuda()
        # print('mask_batch siz : ', msk_batch.size())

        mypad = nn.ConstantPad2d(
            (int(pad + 0.5), int(pad), int(pad + 0.5), int(pad)), 0)

        adv_batch = mypad(adv_batch)
        msk_batch = mypad(msk_batch)  # [batch, max_lab, 3, 608, 608]

        #   保存这里的adv_batch示意
        
        # anglesize = (lab_batch.size(0) * lab_batch.size(1))  #
        anglesize = lab_batch.size(0)
        # anglesize_ = lab_batch.size()[0]
        if do_rotate:
            angle = torch.cuda.FloatTensor(anglesize).uniform_(
                self.minangle, self.maxangle)  # 2
            # 将tensor用从均匀分布中的值填充
        else:
            angle = torch.cuda.FloatTensor(anglesize).fill_(0)

        # Resizes and rotates，调整大小并缩放
        current_patch_size = adv_patch.size(-1)  # 224
        # lab_batch_scaled = torch.cuda.FloatTensor(
        #     lab_batch.size()).fill_(0)  # [batch, max_lab, 5]   #   全 0 tensor
        #   这里学习torch.cuda.FloatTensor()语法
        '''
        # 0.4标签数据
        lab_batch_scaled = torch.cuda.FloatTensor(lab_batch.size(
            0), 1, 5).fill_(0)  # [batch, max_lab, 5]   #   全0tensor
        #   如果场景中没目标，即lab_batch_scaled = [1,1,1,1,1]
        #   以下要对lab_batch进行筛选，只保留最大size的lab
        
        lab_batch_selected = self.lab_transform(lab_batch)  # 调用函数对其进行变换

        #   以下进行还原，用于计算缩放系数
        lab_batch_scaled[:, :, 1] = lab_batch_selected[:,
                                                       :, 1] * img_size  # [batch,1,5]
        lab_batch_scaled[:, :, 2] = lab_batch_selected[:,
                                                       :, 2] * img_size  # 第2、3（1，2）列没用上

        lab_batch_scaled[:, :, 3] = lab_batch_selected[:, :, 3] * img_size
        lab_batch_scaled[:, :, 4] = lab_batch_selected[:, :, 4] * img_size

        #---------------------------------------------------------------#
        #   定义
        #---------------------------------------------------------------#
        pre_scale = SCALE_FACTOR  # 缩放因子
        #   获得缩放因子
        target_size = torch.sqrt(((lab_batch_scaled[:, :, 3].mul(
            1 / pre_scale)) ** 2) + ((lab_batch_scaled[:, :, 4].mul(1 / pre_scale)) ** 2))  # [batch, 1]
        '''
        #   0.01，7列数据格式
        lab_batch_scaled = torch.cuda.FloatTensor(lab_batch.size(
            0), 1, 7).fill_(0)  # [batch, max_lab, 5]   #   全0tensor
        lab_batch_selected = self.lab_transform(lab_batch)  
        # 调用函数对其进行变换，也可以认为是进行选择吧，选择一个参考检测框

        #   以下进行还原，用于计算缩放系数
        lab_batch_scaled[:, :, 0] = lab_batch_selected[:,
                                                       :, 0] * img_size  # [batch,1,5]
        lab_batch_scaled[:, :, 1] = lab_batch_selected[:,
                                                       :, 1] * img_size  # 第2、3（1，2）列没用上

        lab_batch_scaled[:, :, 2] = lab_batch_selected[:, :, 2] * img_size
        lab_batch_scaled[:, :, 3] = lab_batch_selected[:, :, 3] * img_size

        #---------------------------------------------------------------#
        #   定义
        #---------------------------------------------------------------#
        pre_scale = SCALE_FACTOR  # 缩放因子
        #   获得缩放因子
        target_size = torch.sqrt(((lab_batch_scaled[:, :, 2].mul(
            1 / pre_scale)) ** 2) + ((lab_batch_scaled[:, :, 3].mul(1 / pre_scale)) ** 2))  # [batch, 1]
        
        '''
        测试时不需要返回
        patch_center_x = (target_x * img_size).view(-1, 1)
        patch_center_y = (target_y * img_size).view(-1, 1)
        
        patch_center_axis = torch.cat([patch_center_x, patch_center_y], 1)
        '''
        scale = target_size / current_patch_size  # 最终的缩放因子
        scale = scale.view(anglesize)

        s = adv_batch.size()  # [batch,1,3,608,608]
        adv_batch = adv_batch.view(
            s[0] * s[1], s[2], s[3], s[4])  # [batch*max_lab,3,608,608]
        msk_batch = msk_batch.view(
            s[0] * s[1], s[2], s[3], s[4])  # [batch*max_lab,3,608,608]

        sin = torch.sin(angle)
        cos = torch.cos(angle)

        theta1 = torch.cuda.FloatTensor(anglesize, 2, 3).fill_(0)
        # theta = torch.zeros(anglesize, 2, 3)  # simpler # theta默认是一个2x3的矩阵
        #   这个矩阵是旋转、缩放和平移同时操作，但是为了位置不重叠，
        #   1. 先做缩放和旋转，再做平移
        #   2. 先做缩放，再所旋转和平移
        #   theta1：只做旋转和缩放

        theta1[:, 0, 0] = cos / scale
        theta1[:, 0, 1] = sin / scale
        theta1[:, 0, 2] = 0
        theta1[:, 1, 0] = -sin / scale
        theta1[:, 1, 1] = cos / scale
        theta1[:, 1, 2] = 0

        grid = F.affine_grid(theta1, adv_batch.shape)  # 第二个参数设置图片大小
        # 同时形状不变（第二个参数）
        # grid = F.affine_grid(theta, adv_batch.shape)
        adv_batch_t = F.grid_sample(adv_batch, grid)
        msk_batch_t = F.grid_sample(msk_batch, grid)

        #   保存这里只进行了旋转和缩放补丁示意
        #   对补丁进行变换
        # img_patch_after_scale_rotate = adv_batch_t[0,:,:,:]
        # img_patch_after_scale_rotate = transforms.ToPILImage()(img_patch_after_scale_rotate)
        # img_patch_raw_savedir_after_scale_rotate = '_patch_after_rotate_scale.png'
        # img_patch_after_scale_rotate.save(os.path.join("DOTA_creation_test_imgs/patch_location_test", img_patch_raw_savedir_after_scale_rotate))
        #   对msk_batch进行保存
        img_msk_after_scale_rotate = msk_batch_t[0, :, :, :]
        # img_msk_after_scale_rotate = transforms.ToPILImage()(img_msk_after_scale_rotate)
        # img_msk_raw_savedir_after_scale_rotate = '_msk_after_rotate_scale.png'
        # img_msk_after_scale_rotate.save(os.path.join("DOTA_creation_test_imgs/patch_location_test", img_msk_raw_savedir_after_scale_rotate))

        # print("msk size : ", img_msk_after_scale_rotate.size())
        single_channel = img_msk_after_scale_rotate[0, :, :]    #   608x608
        # print("msk after scale and rotate : ", single_channel)
        
        #   得到某一通道所有非0元素，具体值为1的索引，表示补丁现在中心点的位置
        # b_index_temp = torch.nonzero(single_channel == 1)
        b_index = torch.nonzero(single_channel == 1).squeeze()
        #   b_index：[N x 2]，其中N表示其中元素等于1的个数
        # print("index try : ", b_index)
        #   得到宽方向的所有索引，第一列所有元素
        index_W = b_index[:, 0]
        #   分别得到最小和最大索引
        index_W_min = torch.min(index_W)
        index_W_max = torch.max(index_W)
        #   得到补丁外接正方形的半边长
        semi_edge = (index_W_max - index_W_min) / 2
        # print("min index : ", index_W_min, "max index : ", index_W_max)

        #   找到msk_batch_t的界限，得到正方形包围框

        ##########################################################################################
        #   这里需要对粘贴位置进行研究，不能是随机，需要不干涉
        ##########################################################################################

        lab_layout = self.inter_axis_cal(lab_batch, semi_edge, img_size)
        #   当只有一个标签时，这里返回的数据维度是多少？
        #####################################################################################
        #   这里的target_x, target_y就是现在补丁的归一化坐标了吧
        #####################################################################################
        position_available = torch.nonzero(lab_layout == 0)  #   再次找到场景中的零元素，作为可选位置
        # print("length of available position : ", len(position_available))
        # print("all available position : ", position_available)
        
        position_rand = random.randint(0, len(position_available))
        #   对于仅有单个标签的会有问题
        position_final = position_available[position_rand]
        # print("final position : ", position_final)
        target_x = position_final[0] / img_size    
        target_y = position_final[1] / img_size
        
        tx = (-target_x + 0.5) * 2  # 两个数  # 这个2和theta/affine_grid机制相关
        ty = (-target_y + 0.5) * 2  # 和仿射变换的坐标相关，左上角为原点(0.,0.)
        #   向右、向下为负，这里的系数2和affine()函数定义相关
        #   creation attack时，位置要随机，就和lab中的位置信息不再相关

        #   定义theta2，用于平移
        theta2 = torch.cuda.FloatTensor(anglesize, 2, 3).fill_(0)
        theta2[:, 0, 0] = 1
        theta2[:, 0, 1] = 0
        theta2[:, 0, 2] = tx  # 平移参数设置
        # theta2[:, 0, 2] = 0
        theta2[:, 1, 0] = 0
        theta2[:, 1, 1] = 1
        theta2[:, 1, 2] = ty
        # theta2[:, 1, 2] = 0
        grid2 = F.affine_grid(theta2, adv_batch.shape)
        adv_batch_t = F.grid_sample(adv_batch_t, grid2)
        msk_batch_t = F.grid_sample(msk_batch_t, grid2)  # 上下两行需要成对出现

        adv_batch_t = adv_batch_t.view(
            s[0], s[1], s[2], s[3], s[4])  # [batch,max_lab,3,608,608]
        msk_batch_t = msk_batch_t.view(
            s[0], s[1], s[2], s[3], s[4])  # [batch,max_lab,3,608,608]

        # [batch,max_lab,3,608,608] # 000001
        adv_batch_t = torch.clamp(adv_batch_t, 0.0, 1.)
        adv_patch_mask = adv_batch_t * msk_batch_t

        # patch_mask = adv_patch_mask[0,0,:,:,:]
        # patch_mask = transforms.ToPILImage()(patch_mask)
        # patch_mask_savedir = 'patch_mask_01.png'
        # patch_mask.save(os.path.join("framework/mask", patch_mask_savedir))
        
        return adv_patch_mask

class HasSusRGB(nn.Module):
    
    def __init__(self):
        super(HasSusRGB, self).__init__()

    def forward(self, RGB_img):

        '''
        RGB_img: rgb image
        CxHxW
        '''
        r_channel = RGB_img[0, :, :]
        g_channel = RGB_img[1, :, :]
        b_channle = RGB_img[2, :, :]
        
        rg =  r_channel - g_channel
        yb = 0.5 * (r_channel+g_channel) - b_channle
        
        rg_mu = torch.mean(rg)
        yb_mu = torch.mean(yb)
        
        rg_sigma = torch.var(rg)
        yb_sigma = torch.var(yb)
        
        # sigma_loss = math.sqrt(rg_sigma+yb_sigma)
        # mu_loss = math.sqrt(rg_mu**2 + yb_mu**2)
        sigma_loss = torch.sqrt(rg_sigma + yb_sigma)
        mu_loss = torch.sqrt(rg_mu ** 2 + yb_mu ** 2)
        colorful_loss = sigma_loss + 0.3 * mu_loss
        
        return colorful_loss
    
    

if __name__ == '__main__':

    # sys.argv=
    '''
    if len(sys.argv) == 3:
        img_dir = sys.argv[1]
        lab_dir = sys.argv[2]

    else:
        print('Usage: ')
        print('  python load_data.py img_dir lab_dir')
        sys.exit()
    '''
    # img_dir = 'inria/Train/pos'
    # lab_dir = 'inria/Train/pos/yolo-labels'

    img_dir = 'CarData_clean'
    lab_dir = 'CarData_clean/yolo-labels'
    # test_loader = torch.utils.data.DataLoader(InriaDataset(img_dir, lab_dir, shuffle=True),
    #                                           batch_size=3, shuffle=True)

    cfgfile = "cfg/yolov2.cfg"
    weightfile = "weights/yolov2.weights"  # weightfile = "weights/yolov2.weights"
    printfile = "non_printability/30values.txt"

    patch_size = 400

    darknet_model = Darknet(cfgfile)
    darknet_model.load_weights(weightfile)
    darknet_model = darknet_model.cuda()

    max_lab_ = 14
    img_size_ = darknet_model.height

    test_loader = torch.utils.data.DataLoader(
        InriaDataset(
            img_dir,
            lab_dir,
            max_lab_,
            img_size_,
            shuffle=True),
        batch_size=8,
        shuffle=True)
    # 数据集测试
    testiter = iter(test_loader)
    images, labels = testiter.next()
    print("images batch size :", images.size(),
          "labels batch size :", labels.size())
    '''返回的label数据是[batch, max_lab, 5]，
    即在数据加载的时候已经对labels数据进行了处理，统一为max_lab'''
    # print('labels data : ', labels)  # 除了正确标签，其它地方都是1填充
    patch_applier = PatchApplier().cuda()
    patch_transformer = PatchTransformer().cuda()
    prob_extractor = MaxProbExtractor(0, 80, cfgfile).cuda()
    nms_calculator = NPSCalculator(printfile, patch_size)
    total_variation = TotalVariation()
    '''以下代码在utils.do_detect()出现，功能一样
    img = Image.open('data/horse.jpg').convert('RGB')
    img = img.resize((darknet_model.width, darknet_model.height))
    width = img.width
    height = img.height
    img = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
    img = img.view(height, width, 3).transpose(0, 1).transpose(0, 2).contiguous()
    img = img.view(1, 3, height, width)
    img = img.float().div(255.0)
    img = torch.autograd.Variable(img)

    output = darknet_model(img)
    '''
    optimizer = torch.optim.Adam(darknet_model.parameters(), lr=0.0001)

    tl0 = time.time()
    tl1 = time.time()
    for i_batch, (img_batch, lab_batch) in enumerate(test_loader):
        tl1 = time.time()
        print('time to fetch items: ', tl1 - tl0)
        img_batch = img_batch.cuda()
        lab_batch = lab_batch.cuda()
        adv_patch = Image.open(
            'data/horse.jpg').convert('RGB')  # 将这张图片作为patch进行测试
        adv_patch = adv_patch.resize((patch_size, patch_size))
        transform = transforms.ToTensor()
        adv_patch = transform(adv_patch).cuda()
        img_size = img_batch.size(-1)  # 最后一个数，是图片的size
        print('transforming patches')
        t0 = time.time()
        adv_batch_t = patch_transformer.forward(adv_patch, lab_batch, img_size)
        print('applying patches')
        t1 = time.time()
        # 都是直接对batch进行操作，也就是这些类定义在batch模式上
        img_batch = patch_applier.forward(img_batch, adv_batch_t)
        img_batch = torch.autograd.Variable(img_batch)
        img_batch = F.interpolate(
            img_batch, (darknet_model.height, darknet_model.width))  # 对补丁和图片进行插值过渡
        print('running patched images through model')
        t2 = time.time()

        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) or (
                    hasattr(
                        obj, 'data') and torch.is_tensor(
                        obj.data)):
                    try:
                        print(type(obj), obj.size())
                    except BaseException:
                        pass
            except BaseException:
                pass

        print(torch.cuda.memory_allocated())

        output = darknet_model(img_batch)
        print('extracting max probs')
        t3 = time.time()
        max_prob = prob_extractor(output)
        t4 = time.time()
        nms = nms_calculator.forward(adv_patch)
        tv = total_variation(adv_patch)
        print('---------------------------------')
        print('        patch transformation : %f' % (t1 - t0))
        print('           patch application : %f' % (t2 - t1))
        print('             darknet forward : %f' % (t3 - t2))
        print('      probability extraction : %f' % (t4 - t3))
        print('---------------------------------')
        print('          total forward pass : %f' % (t4 - t0))
        del img_batch, lab_batch, adv_patch, adv_batch_t, output, max_prob
        torch.cuda.empty_cache()
        tl0 = time.time()
