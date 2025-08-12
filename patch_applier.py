"""
测试部分
仅是给inria数据集中的测试数据贴上补丁，并没有进一步的检测
本文件下更应该学习的是如何将补丁贴在目标上
"""


import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageDraw
from utils import *
from darknet_v3 import *
# from load_data import PatchTransformer, PatchApplier, InriaDataset

import fnmatch
import copy
import load_data
import utils_self


if __name__ == '__main__':
    # 配置地址
 
    cfgfile = "cfg/yolo-dota.cfg"
    weightfile = "weights/yolov2_pre.weights"

    patchfile = "DOTA_test_data/300.png"
    imgdir = "DOTA_test_data/testing_img"
    # clean_labdir = "suv_photos/clean/yolo-labels"

    savedir = "DOTA_test_data/patch_applier_test"

    n_png_images = len(fnmatch.filter(os.listdir(imgdir), '*.png'))
    n_jpg_images = len(fnmatch.filter(os.listdir(imgdir), '*.jpg'))
    n_images_patched = n_png_images + n_jpg_images  # 应该和n_images_clean一致
    print("total images : ", n_images_patched)

    # 配置模型
    darknet_model = Darknet(cfgfile)
    darknet_model.load_weights(weightfile)
    
    # print("darknet info :", darknet_model)  # debug
    
    darknet_model = darknet_model.eval().cuda()
    patch_applier = load_data.PatchApplier().cuda()
    patch_transformer = load_data.PatchTransformer().cuda()
    # 超参设置
    batch_size = 1
    max_lab = 14
    img_size = darknet_model.height
    img_width = darknet_model.width  # 可直接访问darknet模型中的参数
    print("input image size of yolov2: ", img_size, img_width)  # 模型的输入size

    # patch_size = 300  # 这里可对补丁进行重resize
    # 对patch进行resize操作
    patch_img = Image.open(patchfile).convert('RGB')

    # patch_img.show()
    # tf = transforms.Resize((patch_size, patch_size))  # 补丁可以任意resize()
    # patch_img = tf(patch_img)
    tf = transforms.ToTensor()
    adv_patch_cpu = tf(patch_img)  # 补丁转换到3维tensor
    adv_patch = adv_patch_cpu.cuda()

    print("Pre-setting Done!")
    class_names = load_class_names('data/dota.names')
    # Loop over cleane beelden  # 对干净样本进行loop
    for imgfile in os.listdir(imgdir): 
        if imgfile.endswith('.jpg') or imgfile.endswith('.png'):  # 这一行还必不可少
            print("new image!")
            name = os.path.splitext(imgfile)[0]  # image name w/o extension
            txtname = name + '.txt' 
            txtpath = os.path.abspath(os.path.join(savedir, 'clean/', 'yolo-labels/', txtname))  # 这里只是生成相应文件，还没往里写东西，这个文件只有一份，不同于图片文件
            imgfile = os.path.abspath(os.path.join(imgdir, imgfile))  # 拼接成绝对路径
                
            img = utils_self.load_image_file(imgfile).convert('RGB')
            w, h = img.size
            print("original w = ", w, ", h = ", h)
            if w == h:
                padded_img = img
            else:
                dim_to_pad = 1 if w < h else 2
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
            resize = transforms.Resize((img_size, img_size))    #  这里是否多余？
            padded_img = resize(padded_img)
            padded_img_copy = copy.deepcopy(padded_img)
            boxes_cls = do_detect(darknet_model, padded_img, 0.4, 0.4, True)
            boxes_cls = nms(boxes_cls, 0.4)
                # 生成检测框
            boxes = []  # 定义对特定类别的boxes
            for box in boxes_cls:
                cls_id = box[6]
                if (cls_id == 0): 
                    boxes.append(box)

            clean_pre_name = name + "_pre_clean.png"
            clean_pre_dir = os.path.join(savedir, 'pre_clean/', clean_pre_name)
            plot_boxes(padded_img, boxes, clean_pre_dir, class_names=class_names)

            textfile = open(txtpath, 'w+')  #
            for box in boxes:
                cls_id = box[6]
                if(cls_id == 0):  #
                    x_center = box[0]
                    y_center = box[1]
                    width = box[2]
                    height = box[3]
                    det_conf = box[4]  # 把预测概率打印，实现进一步筛选
                    textfile.write(
                        f'{cls_id} {x_center} {y_center} {width} {height} {det_conf}\n')
            textfile.close()
            

            if os.path.getsize(txtpath):
                    label = np.loadtxt(txtpath)
            else:
                label = np.ones([5])
            label = torch.from_numpy(label).float()
            if label.dim() == 1:
                label = label.unsqueeze(0)
            transform = transforms.ToTensor()
            padded_img = transform(padded_img_copy).cuda()  # 变成了三通道tensor数据

            lab_fake_batch = label.unsqueeze(0).cuda()
            adv_batch_t = patch_transformer(
                adv_patch, lab_fake_batch, img_size, rand_loc=False)
            p_img_batch = patch_applier(padded_img, adv_batch_t)  # [1,3,608,608]

            p_img = p_img_batch.squeeze(0)  # 降维，第一维压缩  [3,608,608]
            p_img_pil = transforms.ToPILImage('RGB')(p_img.cpu())  # 得到添加补丁后图片
            properpatchedname = name + "_p.png"
            p_img_pil.save(os.path.join(
                savedir, 'proper_patched/', properpatchedname))

            txtname = properpatchedname.replace('.png', '.txt')
            txtpath = os.path.abspath(os.path.join(
                savedir, 'proper_patched', 'yolo-labels/', txtname))
            boxes_cls = do_detect(darknet_model, p_img_pil, 0.4, 0.4, True)
            boxes_cls = nms(boxes_cls, 0.4)
            # 次数的det_conf = 0.01，为何这么小？改写成0.4试试
            boxes = []  # 定义对特定类别的boxes
            for box in boxes_cls:
                cls_id = box[6]
                if (cls_id == 0):  # 怎么筛选？
                    boxes.append(box)
            '''增加作框图工作'''
            patched_pre_name = name + "_pre_patched.png"
            patched_pre_dir = os.path.join(savedir, 'pre_patched/', patched_pre_name)
            
            plot_boxes(p_img_pil, boxes, patched_pre_dir, class_names=class_names)
            #   这里不再进行筛选，直接作图


            textfile = open(txtpath, 'w+')
            for box in boxes:
                cls_id = box[6]
                if(cls_id == 0):  #
                    x_center = box[0]
                    y_center = box[1]
                    width = box[2]
                    height = box[3]
                    det_conf = box[4]
                    textfile.write(
                        f'{cls_id} {x_center} {y_center} {width} {height} {det_conf}\n')
                    # patch_results.append({'image_id': name, 'bbox': [x_center.item() - width.item() / 2, y_center.item(
                    # ) - height.item() / 2, width.item(), height.item()], 'score': box[4].item(), 'category_id': 1})
            textfile.close()
print("All Done!")

