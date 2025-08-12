from PIL import Image
from utils import *
from darknet_v3 import Darknet
import fnmatch
import time
import json
import utils_self
from load_data import * 
from torchvision import transforms
import copy
'''
传入补丁，对整个数据集进行测试，在train_patch.py中调用，不更改格式
'''
#-----------------------------------------------------------#
#   在creation attack引入两个指标，分别是
#   1. 增加的instances百分比，越大越好
#   2. 增加的平均obj_conf，越大越好
#   不再使用recall和FR
#-----------------------------------------------------------#


def eval_list(image_dir, label_dir, adv_patch_epoch, cls_ID):


    cfgfile = "cfg/yolov3-dota.cfg"
    weightfile = "/mnt/jfs/tangguijian/DOTA_YOLOv3_patch_AT/weights/yolov3-dota_110000.weights"

    imgdir = image_dir
    labdir = label_dir  # 在训练集

    patch_applier = PatchApplier().cuda()
    patch_transformer = PatchTransformer().cuda()
    conf_thresh = 0.4  # 一开始是0.4

    m = Darknet(cfgfile)
    m.load_darknet_weights(weightfile)
    m.eval()
    eval_wid = m.width
    eval_hei = m.height

    print("the model's width and height is :", m.width, "and ", m.height)
    use_cuda = 1
    if use_cuda:
        m.cuda()

    conf_thresh = conf_thresh  # 最终需要对这个值设置
    nms_thresh = 0.4
    iou_thresh = 0.4  # AP跟iou阈值有关？  # 固定不变
    min_box_scale = 8. / m.width
    # print("min box scale is ", min_box_scale)
    obj_conf = 0.
    total = 0.0
    proposals = 0.0
    correct = 0.0
    lineId = 0
    avg_iou = 0.0
    n_png_images = len(fnmatch.filter(os.listdir(imgdir), '*.png'))
    n_jpg_images = len(fnmatch.filter(os.listdir(imgdir), '*.jpg'))

    n_images = n_png_images + n_jpg_images
    print("total images : ", n_images)

    adv_patch = adv_patch_epoch.cuda()

    for imgfile in os.listdir(imgdir):  # 遍历路径是干净样本
        # print("new image")
        if imgfile.endswith('.jpg') or imgfile.endswith('.png'):
            name = os.path.splitext(imgfile)[0]

            imgfile = os.path.abspath(os.path.join(imgdir, imgfile))
            img = utils_self.load_image_file(imgfile)
            w, h = img.size
            # print("original w = ", w, ", h = ", h)
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
            resize = transforms.Resize((eval_wid, eval_hei))
            padded_img = resize(padded_img)
            # origin_clean_img = copy.deepcopy(padded_img)

            txtname = name + '.txt'  # 打开labels
            lab_path = os.path.abspath(os.path.join(
                labdir, txtname))  # 得到标签路径

            if os.path.getsize(lab_path):  # 文件是否为空，可对写入文件成功与否进行判断
                label = np.loadtxt(lab_path)
            else:
                label = np.ones([5])  # 否则生成全1数字
            label = torch.from_numpy(label).float()  # 将
            if label.dim() == 1:
                label = label.unsqueeze(0)

            transform = transforms.ToTensor()  # 图片转tensor
            # padded_img：表示对图片进行了padding
            padded_img = transform(padded_img).cuda()
            img_fake_batch = padded_img.unsqueeze(0)  # 对单个图片操作，因此这里需要扩维
            lab_fake_batch = label.unsqueeze(0).cuda()

            adv_batch_t, _ = patch_transformer(
                adv_patch, lab_fake_batch, m.height, do_rotate=True, rand_loc=False)  # 设置参数
            p_img_batch = patch_applier(
                img_fake_batch, adv_batch_t)  # 给干净图片添加补丁
            p_img = p_img_batch.squeeze(0)  # 再压缩？
            p_img_pil = transforms.ToPILImage('RGB')(p_img.cpu())

            lineId = lineId + 1  # 共有多少张图片

            truths = read_truths(lab_path)  # 真实标签 
            #   现在真实标签中没有obj_conf

            boxes = do_detect(m, p_img_pil, conf_thresh, nms_thresh, use_cuda)

            boxes_cls = []  # 定义对特定类别的boxes
            for box in boxes:
                cls_id = box[6]
                if (cls_id == cls_ID):  # 怎么筛选？
                    # if (box[2] >= 0.1 and box[3] >= 0.1):     
                    boxes_cls.append(box)
                    obj_conf += box[4]      #   box[4]-->det_conf 


            total = total + truths.shape[0]  # ground_truth labels的个数

    # 简单理解为 recall越低越好

    return precision, recall


if __name__ == '__main__':
    t0 = time.time()

    patchfile = 'testing_patches/testing_out/random_patches/patches/gray_100.png'
    patch_img = utils_self.load_image_file(patchfile)

    img_dir = '/mnt/share1/tangguijian/Data_storage/DOTA_data_01filter/testset'
    lab_dir = '/mnt/share1/tangguijian/Data_storage/DOTA_data_01filter/testset/yolo-labels'
    tf = transforms.ToTensor()
    adv_patch_cpu = tf(patch_img)
    precision, recall = eval_list(img_dir, lab_dir, adv_patch_cpu)
    print("final precision : ", precision)
    print("final recall : ", recall)
    t1 = time.time()
    print('Total running time: {:.4f} minutes'.format((t1 - t0) / 60))

