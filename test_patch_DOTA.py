'''
可以计算recall
可以保存粘贴补丁后图片和检测图片
'''

import sys
import time
import os
import torch

from torchvision import transforms
from PIL import Image
from utils import *
from utils_self import *
from darknet_v3 import *
from load_data import PatchTransformer, PatchApplier, PatchTransformer_test_mode, PatchTransformer_vanishing

import copy
import utils_self


#  # 配置地址
print("Setting everything up")

# imgdir = '/mnt/jfs/tangguijian/Data_storage/DOTA_patch_608/detect_filter_01_label_6/testset'
# clean_labdir = "/mnt/jfs/tangguijian/Data_storage/DOTA_patch_608/detect_filter_01_label_6/testset/yolo-labels"  # 原始标签路径

imgdir = 'framework/imgs'
clean_labdir = "framework/imgs/labels"  # 原始标签路径

cfgfile = "cfg/yolov3-dota.cfg"
weightfile = "/mnt/jfs/tangguijian/DOTA_YOLOv3_patch_AT/weights/yolov3-dota_110000.weights"

patchfile = "framework/transformer/300_patch.png"  #
savedir = "framework"  #


TARGET_ID = 2      #    训练时待攻击的target id

print("patchfile : ", patchfile)
# print("test data : ", imgdir)  # 把测试补丁和测试数据打印记录
print("savedir : ", savedir)
# # 配置模型
darknet_model = Darknet(cfgfile)
darknet_model.load_darknet_weights(weightfile)
print("Matched! Loading weights from ", weightfile)
darknet_model = darknet_model.eval().cuda()

patch_applier = PatchApplier().cuda()
patch_transformer = PatchTransformer().cuda()
# patch_transformer_test_mode = PatchTransformer_test_mode().cuda()

# patch_transformer_vanishing = PatchTransformer_vanishing().cuda()
# 超参设置
batch_size = 1
img_size = darknet_model.height
img_width = darknet_model.width  # 可直接访问darknet模型中的参数
print("input image size of yolov3: ", img_size, img_width)  # 模型的输入size


patch_img = utils_self.load_image_file(patchfile)

tf = transforms.ToTensor()
adv_patch_cpu = tf(patch_img)  # 补丁转换到3维tensor
adv_patch = adv_patch_cpu.cuda()


t0 = time.time()
print("Pre-setting Done!")
class_names = load_class_names('data/dota.names')
# Loop over cleane beelden  # 对干净样本进行loop
for imgfile in os.listdir(imgdir):  # 得到所有文件名
    print("new image")  # 对路径下的图片进行遍历
    if imgfile.endswith('.jpg') or imgfile.endswith('.png'):  # 判断是否为指定文件结尾
        name = os.path.splitext(imgfile)[0]  # image name w/o extension
        # 将文件名分开
        txtname = name + '.txt'  # 将分离出来的文件名重新保存为txt文件
        # txtpath = os.path.abspath(os.path.join(
        #     savedir, 'clean/', 'yolo-labels/', txtname))  # 这里只是生成相应文件，还没往里写东西，这个文件只有一份，不同于图片文件
        txtpath = os.path.abspath(os.path.join(clean_labdir, txtname))

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
        #---------------------------------------------------------------------------#
        # #   对干净样本的预测
        # #---------------------------------------------------------------------------#
        # padded_img_copy = copy.deepcopy(padded_img)
        # boxes_cls = do_detect(darknet_model, padded_img_copy, 0.4, 0.4, True)
        # # print("Here!")
        # boxes_cls = nms(boxes_cls, 0.4)
        #     # 生成检测框
        # boxes = []  # 定义对特定类别的boxes
        # for box in boxes_cls:
        #     cls_id = box[6]
        #     if (cls_id == 0):
        #         if (box[2] >= 0.1 and box[3] >= 0.1):
        #             #   似乎这里也可以不进行筛选，而是选择所有的数据，然后用于计算recall？
        #             boxes.append(box)

        # clean_pre_name = name + "_pre_clean.png"
        # clean_pre_dir = os.path.join(savedir, 'pre_clean/', clean_pre_name)
        # plot_boxes(padded_img_copy, boxes, clean_pre_dir, class_names=class_names)

        # txtpath_pre_clean = os.path.abspath(os.path.join(savedir, 'pre_clean', 'yolo-labels', txtname))
        # textfile = open(txtpath_pre_clean, 'w+')  #
        # for box in boxes:
        #     textfile.write(f'{box[0]} {box[1]} {box[2]} {box[3]} {box[4]} {box[5]} {box[6]}\n')
        # textfile.close()
        
        # padding处理后得到的图片大小和yolov2输入一致
        print("size after padding is ", padded_img.size)

        if os.path.getsize(txtpath):
            label = np.loadtxt(txtpath)
        else:
            label = np.ones([5])
        label = torch.from_numpy(label).float()
        if label.dim() == 1:
            label = label.unsqueeze(0)

        transform = transforms.ToTensor()
        padded_img = transform(padded_img).cuda()  # 变成了三通道tensor数据
        # padded_img = transform(padded_img_copy).cuda()
        img_fake_batch = padded_img.unsqueeze(0)
        # 增维 [1,3,608,608], 此时因为是对图片进行遍历，所以需要扩维，训练时是batch情况
        lab_fake_batch = label.unsqueeze(0).cuda()  # 增维 [1,2,5]

        # transformeer patch en voeg hem toe aan beeld  # 对补丁进行变换并添加到图片中
        adv_batch_t, _ = patch_transformer(
            adv_patch, lab_fake_batch, img_size, do_rotate=True, rand_loc=False)
        # adv_batch_t = patch_transformer_vanishing(
        #     adv_patch, lab_fake_batch, img_size, do_rotate=True, rand_loc=False)
        #   在测试阶段，只需要把补丁粘贴，不需要进行概率提取
        p_img_batch = patch_applier(
            img_fake_batch, adv_batch_t)  # [1,3,608,608]


        p_img = p_img_batch.squeeze(0)  # 降维，第一维压缩  [3,608,608]
        # p_img = F.interpolate(p_img,(darknet_model.height, darknet_model.width))  # 对补丁图片进行了插值
        p_img_pil = transforms.ToPILImage('RGB')(p_img.cpu())  # 得到添加补丁后图片
        
        # properpatchedname = name + ".png"
        # p_img_pil.save(os.path.join(
        #     savedir, 'proper_patched_vanish/', properpatchedname))  # 保存添加补丁图片

        
        txtpath_write = os.path.abspath(os.path.join(
            savedir, 'yolo-labels/', txtname))
        # output_detect = do_detect(darknet_model, p_img_pil, 0.4, 0.4, True)
        # boxes_cls = non_max_suppression(output_detect, 0.4, 0.4)[0]        # 模型输出的值同样经过了softmax和sigmoid处理
        boxes_cls = do_detect(darknet_model, p_img_pil, 0.4, 0.4, True)
        # 次数的det_conf = 0.01，为何这么小？改写成0.4试试
        
        #   以下对检测框目标进行挑选
        # boxes = []  # 定义对特定类别的boxes
        # for box in boxes_cls:
        #     cls_id = box[6]
        #     # if (cls_id == TARGET_ID):  # 怎么筛选？
        #     if (cls_id == TARGET_ID or cls_id == 0):  
        #         #   进一步的，对于攻击的cls_ID和原来ID不一样情况，怎么筛选？
        #         # if (box[2] >= 0.1 and box[3] >= 0.1):
        #         boxes.append(box)
        # '''增加作框图工作'''
        
        patched_pre_name = name + ".png"    #   不添加后缀
        patched_pre_dir = os.path.join(
            savedir, 'pre_patched/', patched_pre_name)

        # plot_boxes(p_img_pil, boxes, patched_pre_dir,
        #            class_names=class_names)  # 这里画出来的预测框也是经过筛选后
        
        
        plot_boxes(p_img_pil, boxes_cls, patched_pre_dir,
                   class_names=class_names)   #   这里使用boxes_cls，表示所有检测框，包括其他类别

        textfile = open(txtpath_write, 'w+')
        for box in boxes_cls:
            textfile.write(f'{box[0]} {box[1]} {box[2]} {box[3]} {box[4]} {box[5]} {box[6]}\n')
        textfile.close()
t1 = time.time()
t_01 = (t1-t0)/60
print("Processing Done!")
print("Total Running Time : ", t_01, "minutes !")

##################################################################################################
#   也可以计算recall
##################################################################################################

# if __name__ == '__main__':
#     t0 = time.time()
#     # cfgfile = "cfg/yolov3-dota.cfg"
#     # weightfile = "weights/yolov3-dota_110000.weights"
#     pre_label_dir = savedir + '/' + 'yolo-labels'
#     # pre_label_dir = 'patches_training_repeat/test_recall_ASR/try_11/yolo-labels'  # 粘贴补丁后的预测结果
#     # pre_label_dir = 'out_recall_test/random_patch_100/pre_patched/yolo-labels'
#     # label不变，为ground truth
#     labdir_gro_tru = '/mnt/jfs/tangguijian/Data_storage/DOTA_patch_608/detect_filter_01_label_6/testset/yolo-labels'

#     n_gro_tru_labels = len(fnmatch.filter(os.listdir(labdir_gro_tru), '*.txt'))
#     n_pre_labels = len(fnmatch.filter(os.listdir(pre_label_dir), '*.txt'))

#     print("total ground truth labels : ", n_gro_tru_labels)
#     print("total predicted labels : ", n_pre_labels)
#     length_gro_tru = txt_len_read(labdir_gro_tru)
#     print("length of ground truth labels : ", length_gro_tru)
#     length_pre_label = txt_len_read(pre_label_dir)
#     print("length of predicted labels : ", length_pre_label)

#     conf_thresh = 0.4  # threshold = 0.4  # 0.4和0.6差别较大
#     precision, recall = utils_self.eval_list(
#         pre_label_dir, labdir_gro_tru, conf_thresh)
#     ASR = (length_gro_tru - length_pre_label) / length_gro_tru
#     print("final precision : ", precision)
#     print("final recall : ", recall)
#     print("final ASR : ", ASR)
#     t1 = time.time()
#     t11 = (t1-t0)/60
#     print('recall & precision Total running time: {:.4f} minutes'.format(t11))
#     # print('recall & precision Total running time: {:.4f} minutes'.format(t11+t_01))
