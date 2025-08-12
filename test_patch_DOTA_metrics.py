'''
在creation attack中，recall指标无法计算，或者说计算无意义
需要引入新的指标
1. instances new-created percentage
2. obj_conf per new-created percentage
3. mAP及AP。这两个概念是一样的，mAP是相较整个测试数据而言，而AP是每个类别的AP变化

引入干涉算法，实现补丁和其它目标尽可能少的干涉

'''

'''
2023-10-30
计算纯图片的攻击效果
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
from load_data import PatchTransformer, PatchApplier, PatchTransformer_vanishing
import load_data
import copy
import utils_self


#  # 配置地址
print("Setting everything up")
print("start training time : ", time.strftime('%Y-%m-%d %H:%M:%S'))

device = torch.device('cuda:0')  # 显卡序号信息
GPU_device = torch.cuda.get_device_properties(device)  # 显卡信息
#   输出显卡参数
print("GPUs : ", GPU_device)

print("scale factor : ", load_data.SCALE_FACTOR)

# imgdir = '/mnt/jfs/tangguijian/DOTA_YOLOv4_patch_creation_AT/trained_patches_test/untargeted_attack/proper_patched'

# imgdir = 'framework/imgs'
# clean_labdir = "framework/imgs/labels"  # 原始标签路径

# cfgfile = "cfg/yolov3-dota.cfg"
# weightfile = "/mnt/jfs/tangguijian/DOTA_YOLOv3_patch_AT/weights/yolov3-dota_110000.weights"

# patchfile = "framework/transformer/300_patch.png"  #
# savedir = "framework"  #
# imgdir = 'trained_patches_test/random_location/for_paper/imgs'
# clean_labdir = 'trained_patches_test/random_location/for_paper/imgs/yolo-labels_w_conf'

#   完整测试集
# imgdir = '/mnt/jfs/tangguijian/Data_storage/creation_patch_attackSet/testset/images'

#   以下是直接用粘贴好的数据进行测试
#   yolov5s
# imgdir = '/mnt/jfs/tangguijian/yolov5_Ultra_creation_DOTA/CreationAttack/trainesPatchesTest/YOLOv5s/untargeted/proper_patched'
#   yolov5x
imgdir = "/mnt/jfs/tangguijian/yolov5_Ultra_creation_DOTA/CreationAttack/trainesPatchesTest/YOLOv5x/untargeted/proper_patched"

ground_labdir = "/mnt/jfs/tangguijian/Data_storage/creation_patch_attackSet/testset" 
clean_labdir = ground_labdir + '/' + 'yolo-labels_w_conf' # 原始标签路径
#   现在把原始的标签作为baseline数据
labdir_gro_tru_thresh_04 = ground_labdir + '/' + 'yolo-labels'  

length_gro_tru_04,_ = txt_len_read(labdir_gro_tru_thresh_04)
print("length of ground truth labels conf--04 : ", length_gro_tru_04)

print("testing imgdir : ", imgdir)
print("testing clean labdir : ", clean_labdir)

n_png_images = len(fnmatch.filter(os.listdir(imgdir), '*.png'))
n_jpg_images = len(fnmatch.filter(os.listdir(imgdir), '*.jpg'))
n_images = n_png_images + n_jpg_images  # 应该和n_images_clean一致
print("Total images : ", n_images)

n_gro_tru = len(fnmatch.filter(os.listdir(clean_labdir), '*.txt'))
#   是ground truth数量
print("total ground truth labels : ", n_gro_tru)
#   分别是ground truth的instances数量
length_gro_tru_label,_ = txt_len_read(clean_labdir)
print("length of ground truth labels conf--001 : ", length_gro_tru_label)
    
    
cfgfile = "cfg/yolov3-dota.cfg"
weightfile = "/mnt/jfs/tangguijian/DOTA_YOLOv3_patch_AT/weights/yolov3-dota_110000.weights"

# patchfile = "transfer_attack/YOLOv5x/220_patch.png"  #
savedir = "transfer_attack/YOLOv5x"  #

# print("patchfile : ", patchfile)
print("savedir : ", savedir)

# # 配置模型
darknet_model = Darknet(cfgfile)
darknet_model.load_darknet_weights(weightfile)
print("Matched! Loading weights from ", weightfile)
darknet_model = darknet_model.eval().cuda()

patch_applier = load_data.PatchApplier().cuda()

# patch_transformer = load_data.PatchTransformer().cuda()
# patch_transformer_vanishing = PatchTransformer_vanishing().cuda()

patch_transformer_test_mode = load_data.PatchTransformer_test_mode(test_mode=True).cuda()
# 超参设置

img_size = darknet_model.height
img_width = darknet_model.width  # 可直接访问darknet模型中的参数
print("input image size of yolov3: ", img_size, img_width)  # 模型的输入size

# patch_img = utils_self.load_image_file(patchfile)
# tf = transforms.ToTensor()
# adv_patch_cpu = tf(patch_img)  # 补丁转换到3维tensor
# adv_patch = adv_patch_cpu.cuda()

t0 = time.time()
print("Pre-setting Done!")
class_names = load_class_names('data/dota.names')

# #   以下直接对粘贴好的补丁进行检测，不需要重新粘贴补丁

for imgfile in os.listdir(imgdir):  # 得到所有文件名
    print("new image")  # 对路径下的图片进行遍历
    if imgfile.endswith('.jpg') or imgfile.endswith('.png'):  # 判断是否为指定文件结尾
        name = os.path.splitext(imgfile)[0]  # image name w/o extension
        # 将文件名分开
        txtname = name + '.txt'  # 将分离出来的文件名重新保存为txt文件
        # 打开图像并调整为yolo输入大小,对每一张图片进行操作
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
        print("size after padding is ", padded_img.size)


        # boxes_cls = do_detect(darknet_model, p_img_pil, 0.4, 0.4, True)
        boxes_cls = do_detect(darknet_model, padded_img, 0.01, 0.4, True)
        #   设置conf_thresh=0.01，得到所有的目标，便于计算mAP和M3
        
        patched_pre_name = name + ".png"    #   不添加后缀
        patched_pre_dir = os.path.join(
            savedir, 'pre_patched/', patched_pre_name)
        
        plot_boxes(padded_img, boxes_cls, patched_pre_dir,
                   class_names=class_names)   #   这里使用boxes_cls，表示所有检测框，包括其他类别

        txtpath_pre = os.path.abspath(os.path.join(
                    savedir, 'yolo-labels', txtname))
        textfile_pre = open(txtpath_pre, 'w+')  #   仅写入筛选后的数据
        
        txtpath_w_conf_pre = os.path.abspath(os.path.join(
                savedir, 'yolo-labels_w_conf', txtname))
        textfile_w_conf_pre = open(txtpath_w_conf_pre, 'w+')    #   写入所有数据，包括置信度信息
         
        for box in boxes_cls:
            textfile_w_conf_pre.write(f'{box[0]} {box[1]} {box[2]} {box[3]} {box[4]} {box[5]} {box[6]}\n')
            obj_conf = box[4]
            if obj_conf > 0.4:
                textfile_pre.write(f'{box[0]} {box[1]} {box[2]} {box[3]} {box[4]} {box[5]} {box[6]}\n')
        textfile_pre.close()
        textfile_w_conf_pre.close()

# #   包括补丁粘贴等完整过程
        
# for imgfile in os.listdir(imgdir):  # 得到所有文件名
#     print("new image")  # 对路径下的图片进行遍历
#     if imgfile.endswith('.jpg') or imgfile.endswith('.png'):  # 判断是否为指定文件结尾
#         name = os.path.splitext(imgfile)[0]  # image name w/o extension
#         # 将文件名分开
#         txtname = name + '.txt'  # 将分离出来的文件名重新保存为txt文件
#         txtpath = os.path.abspath(os.path.join(clean_labdir, txtname))

#         # 打开图像并调整为yolo输入大小,对每一张图片进行操作
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
#         print("size after padding is ", padded_img.size)

#         if os.path.getsize(txtpath):
#             label = np.loadtxt(txtpath)
#         else:
#             # label = np.ones([5])
#             label = np.ones([7])
#         label = torch.from_numpy(label).float()
#         if label.dim() == 1:
#             label = label.unsqueeze(0)

#         transform = transforms.ToTensor()
#         padded_img = transform(padded_img).cuda()  # 变成了三通道tensor数据
#         img_fake_batch = padded_img.unsqueeze(0)
#         # 增维 [1,3,608,608], 此时因为是对图片进行遍历，所以需要扩维，训练时是batch情况
#         lab_fake_batch = label.unsqueeze(0).cuda()  # 增维

#         # transformeer patch en voeg hem toe aan beeld  # 对补丁进行变换并添加到图片中
#         adv_batch_t = patch_transformer_test_mode(
#             adv_patch, lab_fake_batch, img_size, do_rotate=True, rand_loc=False)
        
#         '''
#         #   用于framework制图
#         patch_mask = adv_batch_t[0,0,:,:,:]
#         patch_mask = transforms.ToPILImage()(patch_mask)
#         patch_mask_savedir = name + '.png'
#         patch_mask.save(os.path.join("framework/mask", patch_mask_savedir))
#         '''
#         # adv_batch_t, _ = patch_transformer(
#         #     adv_patch, lab_fake_batch, img_size, do_rotate=True, rand_loc=False)
#         # adv_batch_t = patch_transformer_vanishing(
#         #     adv_patch, lab_fake_batch, img_size, do_rotate=True, rand_loc=False)
#         #   在测试阶段，只需要把补丁粘贴，不需要进行概率提取
#         p_img_batch = patch_applier(
#             img_fake_batch, adv_batch_t)  # [1,3,608,608]
#         p_img = p_img_batch.squeeze(0)  # 降维，第一维压缩  [3,608,608]
#         p_img_pil = transforms.ToPILImage('RGB')(p_img.cpu())  # 得到添加补丁后图片
        
#         # properpatchedname = name + ".png"
#         # p_img_pil.save(os.path.join(
#         #     savedir, 'proper_patched/', properpatchedname))  
#         # 保存添加补丁图片，这一部分需要保存用来开展迁移攻击

#         # boxes_cls = do_detect(darknet_model, p_img_pil, 0.4, 0.4, True)
#         boxes_cls = do_detect(darknet_model, p_img_pil, 0.01, 0.4, True)
#         #   设置conf_thresh=0.01，得到所有的目标，便于计算mAP和M3
        
#         # patched_pre_name = name + ".png"    #   不添加后缀
#         # patched_pre_dir = os.path.join(
#         #     savedir, 'pre_patched/', patched_pre_name)
        
#         # plot_boxes(p_img_pil, boxes_cls, patched_pre_dir,
#         #            class_names=class_names)   #   这里使用boxes_cls，表示所有检测框，包括其他类别

#         txtpath_pre = os.path.abspath(os.path.join(
#                     savedir, 'yolo-labels', txtname))
#         textfile_pre = open(txtpath_pre, 'w+')  #   仅写入筛选后的数据
        
#         txtpath_w_conf_pre = os.path.abspath(os.path.join(
#                 savedir, 'yolo-labels_w_conf', txtname))
#         textfile_w_conf_pre = open(txtpath_w_conf_pre, 'w+')    #   写入所有数据，包括置信度信息
         
#         for box in boxes_cls:
#             textfile_w_conf_pre.write(f'{box[0]} {box[1]} {box[2]} {box[3]} {box[4]} {box[5]} {box[6]}\n')
#             obj_conf = box[4]
#             if obj_conf > 0.4:
#                 textfile_pre.write(f'{box[0]} {box[1]} {box[2]} {box[3]} {box[4]} {box[5]} {box[6]}\n')
#         textfile_pre.close()
#         textfile_w_conf_pre.close()
        
t1 = time.time()
t_01 = (t1-t0)/60
print("Processing Done!")
print("Total Running Time : ", t_01, "minutes !")

# ##################################################################################################
# #   以下计算指标
# ##################################################################################################

if __name__ == '__main__':
    t0 = time.time()
    #   经过conf_thresh=0.4检测数据，用来计算metric-1和metric-2
    pre_label_dir_thresh_04 = savedir + '/' + 'yolo-labels' # 粘贴补丁后的预测结果
    labdir_gro_tru_thresh_04 = ground_labdir + '/' + 'yolo-labels'     #   表示经过了置信度0.4预测检测的数据
    
    #   使用conf_thresh=0.01检测数据，用来计算metric-3
    pre_label_dir_thresh_001 = savedir + '/' + 'yolo-labels_w_conf'
    labdir_gro_tru_thresh_001 = ground_labdir + '/' + 'yolo-labels_w_conf'  
    #   这部分在"DOTA_creation_test_imgs/test_imgs"文件夹下没有
    
    #   M4计算(统计绝对数量似乎没必要，在AP和FP_rate中可以体现)
    instances_per_ID_list_ground_001 = utils_self.instances_per_class_cal(labdir_gro_tru_thresh_001, len(class_names))
    #   统计每个instance的长度，输出的是个和类别长度一致的list，每个类别的数量
    # print("instances distribution per class ", instances_per_ID)
    instances_per_ID_tensor_ground_001 = torch.tensor(instances_per_ID_list_ground_001)   #   将list转换成tensor
    
    
    instances_per_ID_list_pre_001 = utils_self.instances_per_class_cal(pre_label_dir_thresh_001, len(class_names))
    instances_per_ID_tensor_pre_001 = torch.tensor(instances_per_ID_list_pre_001)   #   将list转换成tensor
    instances_gap_M4 = instances_per_ID_tensor_pre_001 - instances_per_ID_tensor_ground_001
    print("isntances gap per class 0.01 (M4) : ", instances_gap_M4)
    
    n_gro_tru_labels = len(fnmatch.filter(os.listdir(labdir_gro_tru_thresh_04), '*.txt'))
    #   实质是图片的数量
    n_pre_labels = len(fnmatch.filter(os.listdir(pre_label_dir_thresh_04), '*.txt'))
    
    #   分别是ground truth和攻击下标签文件数量
    print("total ground truth labels : ", n_gro_tru_labels)
    print("total predicted labels : ", n_pre_labels)
    
    #   分别是ground truth和攻击预测标签的instances数量
    length_gro_tru_04,_ = txt_len_read(labdir_gro_tru_thresh_04)
    print("length of ground truth labels (0.4): ", length_gro_tru_04)
    length_pre_label_04,_ = txt_len_read(pre_label_dir_thresh_04)
    print("length of predicted labels (0.4): ", length_pre_label_04)

    #   metric-1，mAP——需要另外本地单独计算
    
    #   metric-2，INS_created/Ave_INS_created，是用平均新增instance还是sum？
    gap_INS_04 = (length_pre_label_04 - length_gro_tru_04)
    Ave_INS_created_04 = gap_INS_04 / n_gro_tru_labels
    print("INS_gap_thresh_04 : ", gap_INS_04)
    print("average instance created conf_0.4 (M1) : ", Ave_INS_created_04)
    #   metric-3，Ave_CONF_created，稍微复杂些，此时需要置信度信息，因此使用的真实标签和预测标签需要使用conf_thresh=0.01得到
    #   stage-1：分别得到ground-truth和预测数据——yolo-labels_w_conf下的置信度sum
    
    length_gro_tru_001,_ = txt_len_read(labdir_gro_tru_thresh_001)
    print("length of ground truth labels conf--001 : ", length_gro_tru_001)
    length_pre_label_001,_ = txt_len_read(pre_label_dir_thresh_001)
    print("length of predicted labels conf--001: ", length_pre_label_001)
    
    gap_INS_CONF_001 = (length_pre_label_001 - length_gro_tru_001)
    print("INS_gap_thresh_001 : ", gap_INS_CONF_001)
    Ave_INS_created_001 = gap_INS_CONF_001 / n_gro_tru_labels
    print("average instance created conf_0.01 (M1) : ", Ave_INS_created_001)
    
    all_obj_conf_ground_001 = utils_self.per_img_conf_sum(labdir_gro_tru_thresh_001)
    all_obj_conf_pre_001 = utils_self.per_img_conf_sum(pre_label_dir_thresh_001)
    
    print("all objectness confidence in ground : ", all_obj_conf_ground_001, '\n',
        "all objectness confidence in prediction : ", all_obj_conf_pre_001)
    
    Ave_CONF_created_CONF_001 = (all_obj_conf_pre_001 - all_obj_conf_ground_001) / gap_INS_CONF_001
    print("average confidence created (M2, Ave_CONF_created_CONF_001) : ", Ave_CONF_created_CONF_001)
    
    all_obj_conf_ground_04 = utils_self.per_img_conf_sum(labdir_gro_tru_thresh_04)
    all_obj_conf_pre_04 = utils_self.per_img_conf_sum(pre_label_dir_thresh_04)
    
    Ave_CONF_created_CONF_04 = (all_obj_conf_pre_04 - all_obj_conf_ground_04) / gap_INS_04
    print("average confidence created (M2, Ave_CONF_created_CONF_04) : ", Ave_CONF_created_CONF_04)


    t1 = time.time()
    t11 = (t1-t0)/60
    print('Total running time: {:.4f} minutes'.format(t11))
    # print('recall & precision Total running time: {:.4f} minutes'.format(t11+t_01))