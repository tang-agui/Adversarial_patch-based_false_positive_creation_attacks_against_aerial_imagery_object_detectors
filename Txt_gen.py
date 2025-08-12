'''
生成train.txt个val.txt
'''
import os


if __name__ == '__main__':
    # imgdir = '/mnt/share1/tangguijian/Data_storage/split_train_608/images'
    # savedir = 'train.txt'

    imgdir = '/mnt/share1/tangguijian/DOTA_YOLOv3_patch_AT/testing_patches_recall_cal/obj_conf_loss_p224_scale_8_wo_sigmoid/pre_patched'
    savedir = '/mnt/share1/tangguijian/DOTA_YOLOv3_patch_AT/PR_curve/obj_loss_p224_scale_8_wo_sigmoid/val.txt'
    textfile = open(savedir, 'w+')
    for imgfile in os.listdir(imgdir):
        print("new image")

        if imgfile.endswith('.jpg') or imgfile.endswith('.png'):
            imgpath = os.path.abspath(os.path.join(imgdir, imgfile))
        textfile.write(f'{imgpath}\n')
    textfile.close()
    #--------------------------------------------------------------------------#
    #   在计算AP的时候，仅保存图片名称，不需要绝对路径
    #--------------------------------------------------------------------------#
    # imgdir = '/mnt/share1/tangguijian/Data_storage/split_val_608/images'
    # savedir = '/mnt/share1/tangguijian/DOTA_YOLOv2/val_mAP.txt'
    imgdir = '/mnt/share1/tangguijian/DOTA_YOLOv3_patch_AT/testing_patches_recall_cal/obj_conf_loss_p224_scale_8_wo_sigmoid/pre_patched'
    savedir = '/mnt/share1/tangguijian/DOTA_YOLOv3_patch_AT/PR_curve/obj_loss_p224_scale_8_wo_sigmoid/val_mAP.txt'
    textfile = open(savedir, 'w+')
    for imgfile in os.listdir(imgdir):
        print("new image")

        if imgfile.endswith('.jpg') or imgfile.endswith('.png'):
            name = os.path.splitext(imgfile)[0]
        textfile.write(f'{name}\n')
    textfile.close()


# if imgfile.endswith('.jpg'):
#             imgpath = os.path.abspath(os.path.join(imgdir, imgfile))
#             txtname = imgpath.replace('.jpg', '.txt')
# if imgfile.endswith('.png'):
#     imgpath = os.path.abspath(os.path.join(imgdir, imgfile))
#     txtname = imgpath.replace('.png', '.txt')