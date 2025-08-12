'''
YOLOv3-DOTA
对干净图片进行预测
2021-12-15：也可对任意图片进行预测
'''

import sys
import time
import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageDraw
from utils import *
from utils_self import *
from darknet_v3 import *
# from load_data import PatchTransformer, PatchApplier, InriaDataset
import json
import copy
import utils_self


 # 配置地址
print("Setting everything up")

imgdir = "trained_patches_test/physical_test/salient_patches/v2_raw"

cfgfile = "cfg/yolov3-dota.cfg"
weightfile = "/mnt/jfs/tangguijian/DOTA_YOLOv3_patch_AT/weights/yolov3-dota_110000.weights"

savedir = "trained_patches_test/physical_test/salient_patches/v2_raw_pre"  #

print("savedir : ", savedir)
# # 配置模型
darknet_model = Darknet(cfgfile)
darknet_model.load_darknet_weights(weightfile)
print("Matched! Loading weights from ", weightfile)
darknet_model = darknet_model.eval().cuda()
img_size = darknet_model.height
img_width = darknet_model.width  # 可直接访问darknet模型中的参数
print("input image size of yolov3: ", img_size, img_width)  # 模型的输入size


print("Pre-setting Done!")
class_names = load_class_names('data/dota.names')
print("length of class_names : ", len(class_names))
# Loop over cleane beelden  # 对干净样本进行loop
t0 = time.time()
for imgfile in os.listdir(imgdir):  # 得到所有文件名

    if imgfile.endswith('.jpg') or imgfile.endswith('.png'):  # 判断是否为指定文件结尾
        print("new image")  # 对路径下的图片进行遍历
        name = os.path.splitext(imgfile)[0]  # image name w/o extension
        # print(os.path.splitext(imgfile))  #输出的是文件名和扩展名，[0]值需要文件名
        # 将文件名分开
        txtname = name + '.txt'  # 将分离出来的文件名重新保存为txt文件

        # open beeld en pas aan naar yolo input size  # 打开图像并调整为yolo输入大小,对每一张图片进行操作
        imgfile = os.path.abspath(os.path.join(imgdir, imgfile))  # 拼接成绝对路径
        print("image file path is ", imgfile)
        img = load_image_file(imgfile)  # 注意这里的不同

        w, h = img.size
        print("original w = ", w, ", h = ", h)
        if w == h:
            padded_img = img
        else:
            dim_to_pad = 1 if w < h else 2   # dim_to_pad = 1 if w < h else 2  # 根据原始图片的宽高决定怎么填充
            if dim_to_pad == 1:
                padding = (h - w) / 2
                padded_img = Image.new(
                    'RGB', (h, h), color=(127, 127, 127))
                padded_img.paste(img, (int(padding), 0))
            else:
                padding = (w - h) / 2
                padded_img = Image.new(
                    'RGB', (w, w), color=(127, 127, 127))
                padded_img.paste(img, (0, int(padding)))

        resize = transforms.Resize((img_size, img_size))
        padded_img = resize(padded_img)
        w_, h_ = padded_img.size
        print("after padding w = ", w_, ", h = ", h_)

        # boxes_cls = do_detect(darknet_model, padded_img, 0.4, 0.4, True)
        boxes_cls = do_detect(darknet_model, padded_img, 0.4, 0.4, True)   #   
        #   这里的boxes格式是[x,y,w,h,obj_conf,cls_conf,cls_id]

        clean_pre_name = name + "_pre_clean.png"
        clean_pre_dir = os.path.join(savedir,  clean_pre_name)  #   'clean_pre/',
        plot_boxes(padded_img, boxes_cls, clean_pre_dir, class_names=class_names)
        
        txtpath_pre_clean = os.path.abspath(
            os.path.join(savedir, 'yolo-labels', txtname))
        textfile = open(txtpath_pre_clean, 'w+')  #
        
        # for box in boxes:
        #     x_center = box[0]
        #     y_center = box[1]
        #     width = box[2]
        #     height = box[3]
        #     obj_conf = box[4]
        #     textfile.write(
        #         f'{box[6]} {obj_conf} {x_center} {y_center} {width} {height}\n')
        #     # textfile.write(f'{box[6]} {box[0]} {box[1]} {box[2]} {box[3]} \n')
        #     #   对于训练样本和测试样本而言，就只需要(id,x,y,w,h)
        # textfile.close()
        
        for box in boxes_cls:
            textfile.write(f'{box[0]} {box[1]} {box[2]} {box[3]} {box[4]} {box[5]} {box[6]}\n')
        textfile.close()


t1 = time.time()
t_01 = (t1-t0)/60
print("Processing Done!")
print("Total Running Time : ", t_01, "minutes !")


# #   以下在使用黑盒攻击算法时，需要重新生成ground truth标签，包含类别信息
# print("Setting everything up")

# imgdir = "/mnt/share1/tangguijian/Data_storage/large_vehicle/wo_L_vehicle_0.1filter/testset_100/img"

# cfgfile = "cfg/yolov3-dota.cfg"
# weightfile = "/mnt/jfs/tangguijian/DOTA_YOLOv3_patch_AT/weights/yolov3-dota_110000.weights"
# savedir = "/mnt/share1/tangguijian/Data_storage/large_vehicle/wo_L_vehicle_0.1filter/testset_100/img_pre"  #

# # print("savedir : ", savedir)
# # # 配置模型
# darknet_model = Darknet(cfgfile)
# darknet_model.load_darknet_weights(weightfile)
# print("Matched! Loading weights from ", weightfile)
# darknet_model = darknet_model.eval().cuda()
# img_size = darknet_model.height
# img_width = darknet_model.width  # 可直接访问darknet模型中的参数
# print("input image size of yolov3: ", img_size, img_width)  # 模型的输入size


# print("Pre-setting Done!")
# class_names = load_class_names('data/dota.names')
# print("length of class_names : ", len(class_names))
# # Loop over cleane beelden  # 对干净样本进行loop
# t0 = time.time()
# for imgfile in os.listdir(imgdir):  # 得到所有文件名

#     if imgfile.endswith('.jpg') or imgfile.endswith('.png'):  # 判断是否为指定文件结尾
#         print("new image")  # 对路径下的图片进行遍历
#         name = os.path.splitext(imgfile)[0]  # image name w/o extension
#         # print(os.path.splitext(imgfile))  #输出的是文件名和扩展名，[0]值需要文件名
#         # 将文件名分开
#         txtname = name + '.txt'  # 将分离出来的文件名重新保存为txt文件
#         # txtname = txtname.replace('_p.txt', '.txt')
#         # txtpath = os.path.abspath(os.path.join(
#         #     savedir, 'clean/', 'yolo-labels/', txtname))  # 这里只是生成相应文件，还没往里写东西，这个文件只有一份，不同于图片文件

#         # open beeld en pas aan naar yolo input size  # 打开图像并调整为yolo输入大小,对每一张图片进行操作
#         imgfile = os.path.abspath(os.path.join(imgdir, imgfile))  # 拼接成绝对路径
#         print("image file path is ", imgfile)
#         img = load_image_file(imgfile)  # 注意这里的不同

#         w, h = img.size
#         print("original w = ", w, ", h = ", h)
#         if w == h:
#             padded_img = img
#         else:
#             dim_to_pad = 1 if w < h else 2   # dim_to_pad = 1 if w < h else 2  # 根据原始图片的宽高决定怎么填充
#             if dim_to_pad == 1:
#                 padding = (h - w) / 2
#                 padded_img = Image.new(
#                     'RGB', (h, h), color=(127, 127, 127))
#                 padded_img.paste(img, (int(padding), 0))
#             else:
#                 padding = (w - h) / 2
#                 padded_img = Image.new(
#                     'RGB', (w, w), color=(127, 127, 127))
#                 padded_img.paste(img, (0, int(padding)))

#         resize = transforms.Resize((img_size, img_size))
#         padded_img = resize(padded_img)
#         w_, h_ = padded_img.size
#         print("after padding w = ", w_, ", h = ", h_)

#         boxes_cls = do_detect(darknet_model, padded_img, 0.4, 0.4, True)
#         # boxes_cls = do_detect(darknet_model, padded_img, 0.01, 0.4, True)   #   
#         #   这里的boxes格式是[x,y,w,h,obj_conf,cls_conf,cls_id]

#         boxes = []  # 定义对特定类别的boxes
#         for box in boxes_cls:
#             cls_id = box[6]
#             if (cls_id == 5):  # 
#                 if (box[2] >= 0.1 and box[3] >= 0.1):
#                     boxes.append(box)

#         clean_pre_name = name + "_pre_clean.png"
#         clean_pre_dir = os.path.join(savedir,  clean_pre_name)
#         plot_boxes(padded_img, boxes, clean_pre_dir, class_names=class_names)

#         txtpath_pre_clean = os.path.abspath(
#             os.path.join(savedir, 'yolo-labels', txtname))
#         textfile = open(txtpath_pre_clean, 'w+')  #
        
#         # for box in boxes:
#         #     x_center = box[0]
#         #     y_center = box[1]
#         #     width = box[2]
#         #     height = box[3]
#         #     obj_conf = box[4]
#         #     textfile.write(
#         #         f'{box[6]} {obj_conf} {x_center} {y_center} {width} {height}\n')
#         #     # textfile.write(f'{box[6]} {box[0]} {box[1]} {box[2]} {box[3]} \n')
#         #     #   对于训练样本和测试样本而言，就只需要(id,x,y,w,h)
#         # textfile.close()
        
#         for box in boxes:
#             textfile.write(f'{box[0]} {box[1]} {box[2]} {box[3]} {box[4]} {box[5]} {box[6]}\n')
#         textfile.close()

# t1 = time.time()
# t_01 = (t1-t0)/60
# print("Processing Done!")
# print("Total Running Time : ", t_01, "minutes !")
