'''
分别对：
- /mnt/jfs/tangguijian/Data_storage/split_train_608
- /mnt/jfs/tangguijian/Data_storage/split_val_608
进行挑选，分别用于creation attack的trainset和testset
'''

import os
import shutil
from torchvision import transforms
from PIL import Image, ImageDraw
from utils import *
from darknet_v3 import *
import utils_self
import time
import fnmatch


#  # 配置地址
print("Setting everything up")
# imgdir = "/mnt/share1/tangguijian/Data_storage/DOTA_patch_608/train_plane_filter"
# # imgdir = "/mnt/share1/tangguijian/Data_storage/DOTA_patch_608/val_plane_filter"

imgdir = "/mnt/jfs/tangguijian/Data_storage/creation_patch_attackSet/trainset_raw/images"
# imgdir = "/mnt/share1/tangguijian/Data_storage/large_vehicle/valset_large_vehicle_filter"
print("imgdir : ", imgdir)
cfgfile = "cfg/yolov3-dota.cfg"
weightfile = "/mnt/jfs/tangguijian/DOTA_YOLOv3_patch_AT/weights/yolov3-dota_110000.weights"

savedir = "/mnt/jfs/tangguijian/Data_storage/creation_patch_attackSet/trainset"  #
'''
在savedir路径下的clean和yolo-labels下分别筛选后的保存图片和标签
'''

n_png_images = len(fnmatch.filter(os.listdir(imgdir), '*.png'))
n_jpg_images = len(fnmatch.filter(os.listdir(imgdir), '*.jpg'))
n_images = n_png_images + n_jpg_images  # 应该和n_images_clean一致
print("Total images : ", n_images)

# 配置模型
darknet_model = Darknet(cfgfile)
darknet_model.load_darknet_weights(weightfile)
print("Matched! Loading weights from ", weightfile)
darknet_model = darknet_model.eval().cuda()

# 超参设置
batch_size = 1
img_size = darknet_model.height
img_width = darknet_model.width  # 可直接访问darknet模型中的参数

print("Pre-setting Done!")
class_names = load_class_names('data/dota.names')
num_val = 0
t0 = time.time()
# Loop over cleane beelden  # 对干净样本进行loop
for imgfile in os.listdir(imgdir):  # 得到所有文件名
    print("new image")  # 对路径下的图片进行遍历
    if imgfile.endswith('.jpg') or imgfile.endswith('.png'):  # 判断是否为指定文件结尾
        name = os.path.splitext(imgfile)[0]  # image name w/o extension

        txtname = name + '.txt'  # 将分离出来的文件名重新保存为txt文件
        # print(txtname, "and", txtpath)
        # open beeld en pas aan naar yolo input size  # 打开图像并调整为yolo输入大小,对每一张图片进行操作
        imgfile = os.path.abspath(os.path.join(imgdir, imgfile))  # 拼接成绝对路径
        img = utils_self.load_image_file(imgfile)  # 注意这里的不同

        w, h = img.size
        print("original w = ", w, ", h = ", h)
        if w == h:
            padded_img = img
        else:
            dim_to_pad = 1 if w < h else 2  # dim_to_pad = 1 if w < h else 2  # 根据原始图片的宽高决定怎么填充
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

        resize = transforms.Resize((img_size, img_size))  # 不管多大，在这里resize
        padded_img = resize(padded_img)
        # padding处理后得到的图片大小和yolov2输入一致


        boxes_cls = do_detect(darknet_model, padded_img, 0.01, 0.4, True)
        #   计算mAP时需要设置obj_conf为较小值
        #   这里的boxes格式是[x,y,w,h,obj_conf,cls_conf,cls_id]
        # boxes = []  # 定义对特定类别的boxes
        # for box in boxes_cls:
        #     cls_id = box[6]
        #     if (cls_id == 5):   #   id==6表示ship
        #         if (box[2] >= 0.1 and box[3] >= 0.1):
        #             boxes.append(box)

        if (len(boxes_cls) > 0):
        #   if (len(boxes))
            num_val+= 1
            txtpath = os.path.abspath(os.path.join(
                    savedir, 'yolo-labels', txtname))
            textfile = open(txtpath, 'w+')  # open打开文件，后面的为具体的模式
            cleanname = name + ".png"
            padded_img.save(os.path.join(savedir, 'images', cleanname))  # 这里开始保存图片，
            
            txtpath_w_conf = os.path.abspath(os.path.join(
                    savedir, 'yolo-labels_w_conf', txtname))
            textfile_w_conf = open(txtpath_w_conf, 'w+')  
            #   textfile和textfile_w_conf两个文件写入东西不一样
            for box in boxes_cls:  # 根据检测框中的w、h进行筛选，并保存图片
                obj_conf = box[4]
                if obj_conf > 0.4:
                    textfile.write(
                        f'{box[6]} {box[0]} {box[1]} {box[2]} {box[3]} \n'  
                        #   保存，不含置信度，且需要对置信度进行筛选，然后作为后面攻击的ground truth数据
                        #   id, x, y, w, h
                        #   从而两个label文件保存数据量不一样
                        )
                textfile_w_conf.write(
                    f'{box[0]} {box[1]} {box[2]} {box[3]} {box[4]} {box[5]} {box[6]}\n'
                    #   保存，包含置信度
                    )
                    
            textfile.close()
            textfile_w_conf.close()
            
print("Total %i images collected! " % (num_val))
t1 = time.time()
print('Total running time: {:.4f} minutes'.format((t1 - t0) / 60))

print("ALL DONE!")

#----------------------------------------------------------------#
#   以下对数据进行训练集、测试集划分
#----------------------------------------------------------------#

# imgdir = "/mnt/jfs/tangguijian/Data_storage/split_train_608"
# desdir = "/mnt/jfs/tangguijian/Data_storage/creation_patch_attackSet/trainset"  #
# num = 0
# # Loop over cleane beelden  # 对干净样本进行loop
# for imgfile in os.listdir(imgdir):  # 得到所有文件名
#     print("new image")  # 对路径下的图片进行遍历
#     if imgfile.endswith('.jpg') or imgfile.endswith('.png'):  # 判断是否为指定文件结尾
#         name = os.path.splitext(imgfile)[0]  # image name w/o extension
#         txtname = name + '.txt'  # 将分离出来的文件名重新保存为txt文件
#         cleanname = name + ".png"

#         txtpath = os.path.abspath(os.path.join(
#             imgdir, 'yolo-labels/', txtname))
#         imgpath = os.path.abspath(os.path.join(imgdir, cleanname))  # 这里开始保存图片，
        
#         '''
#         des_txtpath = os.path.abspath(
#             os.path.join(desdir, 'yolo-labels/', txtname))
#         des_imgpath = os.path.abspath(
#             os.path.join(desdir, 'img/', cleanname))  # 这里开始保存图片，

#         shutil.copy(txtpath, des_txtpath)
#         shutil.copy(imgpath, des_imgpath)
#         '''
#         #   对于这种复制操作，des_xxx应该可以不需要指定文件名，指定文件路径即可
#         #   
#         des_txtpath = os.path.abspath(
#             os.path.join(desdir, 'yolo-labels/'))
#         des_imgpath = os.path.abspath(
#             os.path.join(desdir, 'img/'))  # 这里开始保存图片，

#         shutil.copy(txtpath, des_txtpath)   #   直接将原文件复制到des路径下
#         shutil.copy(imgpath, des_imgpath)

#         if num > 98:
#             break
#         num += 1
# print("ALL DONE!")
