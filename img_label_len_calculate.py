import utils_self
import fnmatch
import os
import torch
import numpy as np


# imgdir = '/mnt/share1/tangguijian/Data_storage/DOTA_patch_trainset/clean'  
# labdir = '/mnt/share1/tangguijian/Data_storage/DOTA_patch_trainset/clean/yolo-labels'  # 到labels层

# imgdir = '/mnt/jfs/tangguijian/Data_storage/creation_patch_attackSet/testset/images'  
# labdir = '/mnt/jfs/tangguijian/Data_storage/creation_patch_attackSet/testset/yolo-labels'  # 到labels层

imgdir = '/mnt/jfs/tangguijian/Data_storage/DOTA_datasets/split_train_608/images'  
labdir = '/mnt/jfs/tangguijian/Data_storage/DOTA_datasets/split_train_608/labels'  # 到labels层

n_png_images = len(fnmatch.filter(os.listdir(imgdir), '*.png'))
n_jpg_images = len(fnmatch.filter(os.listdir(imgdir), '*.jpg'))
n_images = n_png_images + n_jpg_images  # 应该和n_images_clean一致
print("Total images : ", n_images)

n_lab = len(fnmatch.filter(os.listdir(labdir), '*.txt'))

length_txt_boxes, len_ins_account = utils_self.txt_len_read(labdir)
#   分别返回总instance和每个文件的instance
# index = (len_ins_account[:] > 100)
# num_instances = np.array(len_ins_account)
# np.save('instances_account/num_instances.npy', num_instances)
# utils_self.hist_draw(len_ins_account,"instances_account/instances_account_01.png")
# filename = open('instances_account/num_instances.txt', 'w')
# a = sum(i <= 6 for i in len_ins_account)
# print("a = ", a)
# for value in len_ins_account:
#     filename.write(str(value))
#     filename.write('\n')
# filename.close()
#------------------------------------------------------------#
#   还需要得到最大的labels
#------------------------------------------------------------#
print("Total instances : ", length_txt_boxes)
print("max length of labels : ", max(len_ins_account))
# print("num of instances per image : ", len_ins_account)
print("Total txt file : ", n_lab)

n_device = torch.cuda.device_count()
# print("num of cudas : ", n_device)



# def diff_txt_filter(labdir):
#     # 用来找到两个文件夹中不一样的文件
#     label_dir_thresh_04 = labdir + '/' + 'yolo-labels' # 粘贴补丁后的预测结果
#     labdir_gro_tru_thresh_001 = labdir + '/' + 'yolo-labels_w_conf'     #   表示经过了置信度0.4预测检测的数据
    
#     for txtfile_label in os.listdir(labdir_gro_tru_thresh_001):  # 得到所有文件名
#         txtfile_001 = os.path.abspath(os.path.join(labdir_gro_tru_thresh_001, txtfile_label)) 
#         txtfile_04 = os.path.abspath(os.path.join(label_dir_thresh_04,txtfile_label))
#         if not os.path.exists(txtfile_04):
#             print(txtfile_001)
#             break
        
#     return 1

# labdir = '/mnt/jfs/tangguijian/Data_storage/creation_patch_attackSet/testset'
# find_diff = diff_txt_filter(labdir)