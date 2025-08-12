
from PIL import Image
import PIL.Image
import PIL.ImageOps
from utils import *
from darknet_v3 import Darknet
import fnmatch
import numpy as np
import matplotlib.pyplot as plt


def eval_list(pre_patched_labels_dir, labdir, conf_thresh):

    conf_thresh = conf_thresh  # 最终需要对这个值设置
    iou_thresh = 0.5  # AP跟iou阈值有关？
    total = 0.0
    proposals = 0.0
    correct = 0.0
    lineId = 0
    avg_iou = 0.0

    for imgfile in os.listdir(pre_patched_labels_dir):  # 预测标签
        # print("new image")
        if imgfile.endswith('.txt'):
            pre_labeld_dir = os.path.abspath(os.path.join(pre_patched_labels_dir, imgfile))
            lab_path = os.path.abspath(os.path.join(
                labdir, imgfile))
            lineId = lineId + 1  # 共有多少张图片

            # replace(oldvalue, newvalue)
            # print("lab_path : ", lab_path)
            truths = read_truths(lab_path)  # 真实标签  # 加载标签
            boxes_cls = read_truths_pre_7(pre_labeld_dir)

            # print("length of boxes_cls :",  len(boxes))
            if False:
                savename = "tmp/%06d.jpg" % (lineId)
                print("save %s" % savename)
                plot_boxes(img, boxes, savename)
            total = total + truths.shape[0]  # ground_truth labels的个数
            # 为真实的正例，是TP + FN
            # print("length of boxes_cls : ", len(boxes_cls))
            for i in range(len(boxes_cls)):
                if (boxes_cls[i][4]*boxes_cls[i][5]) > conf_thresh:  # 第五个元素为检测obj概率
                    proposals = proposals+1  # 预测出来的检测框数量
                    # proposal即为预测为正值的情况，为TP+FP
            for i in range(truths.shape[0]):
                box_gt = [truths[i][1], truths[i][2],
                          truths[i][3], truths[i][4], 1.0]
                best_iou = 0
                for j in range(len(boxes_cls)):
                    iou = bbox_iou(
                        box_gt, boxes_cls[j], x1y1x2y2=False)  # 计算IoU
                    best_iou = max(iou, best_iou)  # 最大的IoU
                if best_iou > iou_thresh:
                    avg_iou += best_iou
                    correct = correct+1  # correct为预测矩阵中的TP，为recall和precision中的分子

    precision = correct/(proposals + 1e-8)  # 对所有图片的结果 TP/(TP+FP)
    recall = correct/(total + 1e-8)  # TP/(TP+FN)
    # 简单理解为 recall越低越好
    fscore = 2.0*precision*recall/(precision+recall + 1e-6)
    # print("results in recall.py :")
    # print("%d IOU: %f, Recal: %f, Precision: %f, Fscore: %f\n" %
    #       (lineId-1, avg_iou/(correct + 1e-6), recall, precision, fscore))
    # print("total images = ", n_images)
    return precision, recall


def ap_calculation(recall, precision, use_07_metric=False):
    '''
    AP calculate
    给定recall和precision的numpy数据，计算此时的AP
    '''
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).(即此时采用的方法其实就是voc_ap中的方法)
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(recall >= t) == 0:
                p = 0
            else:
                p = np.max(precision[recall >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], recall, [1.]))
        mpre = np.concatenate(([1.], precision, [0.]))  #
        '''print("mrec : ", mrec, "mpre : ", mpre)
        mpre_sorted = sorted(mpre, reverse=True)
        print("mpre_sorted : ", mpre_sorted)'''
        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i-1] = np.maximum(mpre[i-1], mpre[i])  # （不完全）相当于排序

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]
        
        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i+1] - mrec[i])* mpre[i+1])

    return ap


def exif_transpose(img):
    if not img:
        return img

    exif_orientation_tag = 274

    # Check for EXIF data (only present on some files)
    if hasattr(img, "_getexif") and isinstance(img._getexif(), dict) and exif_orientation_tag in img._getexif():
        exif_data = img._getexif()
        orientation = exif_data[exif_orientation_tag]

        # Handle EXIF Orientation
        if orientation == 1:
            # Normal image - nothing to do!
            pass
        elif orientation == 2:
            # Mirrored left to right
            img = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)
        elif orientation == 3:
            # Rotated 180 degrees
            img = img.rotate(180)
        elif orientation == 4:
            # Mirrored top to bottom
            img = img.rotate(180).transpose(PIL.Image.FLIP_LEFT_RIGHT)
        elif orientation == 5:
            # Mirrored along top-left diagonal
            img = img.rotate(-90, expand=True).transpose(PIL.Image.FLIP_LEFT_RIGHT)
        elif orientation == 6:
            # Rotated 90 degrees
            img = img.rotate(-90, expand=True)
        elif orientation == 7:
            # Mirrored along top-right diagonal
            img = img.rotate(90, expand=True).transpose(PIL.Image.FLIP_LEFT_RIGHT)
        elif orientation == 8:
            # Rotated 270 degrees
            img = img.rotate(90, expand=True)

    return img


def load_image_file(file, mode='RGB'):
    # Load the image with PIL
    img = PIL.Image.open(file)

    if hasattr(PIL.ImageOps, 'exif_transpose'):
        # Very recent versions of PIL can do exit transpose internally
        img = PIL.ImageOps.exif_transpose(img)
    else:
        # Otherwise, do the exif transpose ourselves
        img = exif_transpose(img)

    img = img.convert(mode)

    return img

def txt_len_read(txtfile_list):
    # for instances calculate
    len_txt = 0
    len_ins_account = []
    for txtfile_label in os.listdir(txtfile_list):  # 得到所有文件名
        txtfile = os.path.abspath(os.path.join(txtfile_list, txtfile_label)) 
        if os.path.getsize(txtfile):
            myfile = open(txtfile)
            single_len = len(myfile.readlines())
            len_txt += single_len
            len_ins_account.append(single_len)

    return len_txt, len_ins_account

def per_img_conf_sum(labels):
    '''
    输入: 0.01置信度阈值下的（真实/预测）标签
    '''
    conf_sum = 0.
    for txtfile_label in os.listdir(labels):
        if txtfile_label.endswith('.txt'):
            txtfile = os.path.abspath(os.path.join(labels, txtfile_label))
            if os.path.getsize(txtfile):
                myfile = open(txtfile)
                file_items = myfile.readlines()     #   这样读进来的格式类似list
                if len(file_items):
                    for item in file_items:
                        # print("per line item : ", item.rsplit())
                        # print("per obj_conf item : ", item.rsplit()[4])
                        conf_sum += float(item.rsplit()[4])
    return conf_sum

from torchvision import transforms
from PIL import Image
import torch

# patchfile_0 = "/mnt/sunjialiang/kitti_car/KITTI_DataSet_mix/easy_exp_1200_3/1880_patch.png"
# patchfile_1= "/mnt/sunjialiang/kitti_car/KITTI_DataSet_mix/easy_exp_1200_3/1810_patch.png"

def patch_MSE_calsulator(patchfile_0, patchfile_1):
# def patch_MSE_calsulator(adv_patch_0, adv_patch_1):
    
    # 计算两个补丁间的MSE
    patch_img_0 = Image.open(patchfile_0).convert('RGB')
    patch_img_1 = Image.open(patchfile_1).convert('RGB')

    tf = transforms.ToTensor()
    adv_patch_0 = tf(patch_img_0)  # 补丁转换到3维tensor
    adv_patch_1 = tf(patch_img_1)

    patch_diff = adv_patch_1 - adv_patch_0
    mse_F = torch.nn.MSELoss()
    patch_mse = mse_F(adv_patch_0, adv_patch_1)  # 计算两个补丁的MSE

    return patch_mse

def hist_draw(data_list, save_dir):
    '''绘制柱状图，根据每张图片统计的instances'''
    plt.bar(range(len(data_list)), data_list)
    plt.xlabel("number of instances")
    plt.ylabel("number of images")
    plt.savefig(save_dir)
    plt.show()

def instances_per_class_cal(labels_dir, num_class):
    '''
    给定文件路径，统计每个类别的instances数量
    labels_dir：
    '''
    ID_list = []    #   用于存放每个instances的ID
    for txtfile_label in os.listdir(labels_dir):
        if txtfile_label.endswith('.txt'):
            txtfile = os.path.abspath(os.path.join(labels_dir, txtfile_label))
            if os.path.getsize(txtfile):    #   不为空
                myfile = open(txtfile)
                file_items = myfile.readlines()
                for item in file_items:
                    id = int(item.rsplit()[-1])  #   最后的元素为
                    '''
                    if id == 0:
                    ...
                    '''
                    ID_list.append(id)
    '''
    接着计算，得到每个类别的数量
    '''
    instances_per_ID = []
    for i in range(num_class):
        instances_len = ID_list.count(i)
        instances_per_ID.append(instances_len)
    
    return instances_per_ID

if __name__ == '__main__':
    '''
    test for ap_calculation
    '''
    # recall = [0.3, 0.6, 0.7, 0.8, 0.9, 0.92]
    # precision = [1.0, 0.95, 0.91, 0.89, 0.79, 0.4]
    #
    # # average_precision = ap_calculation(recall, precision)
    # # print("calculated AP :", average_precision)
    # patch_mse = patch_MSE_calsulator(patchfile_0, patchfile_1)
    # print("patch_mse :", patch_mse)
    # num_list = [1.5, 0.6, 7.8, 6]
    # hist_draw(num_list, "/mnt/share1/tangguijian/DOTA_YOLOv3_patch_AT/instances_account/")

    test_labdir = 'DOTA_creation_test_imgs/Ave_CONF_created'
    sum_conf = per_img_conf_sum(test_labdir)
    print("sum of conf per image : ", sum_conf)