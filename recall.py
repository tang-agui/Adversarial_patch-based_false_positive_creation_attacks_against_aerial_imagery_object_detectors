from PIL import Image, ImageDraw
from utils import *
from darknet import Darknet

'''
在''test_recall.txt'文件中给出待recall计算的目标图片
可直接运行
'''

def eval_list(cfgfile, weightfile, imglist):
    # imglist是什么数据格式？
    #m = TinyYoloFace14Net()
    # m.eval()
    # m.load_darknet_weights(tiny_yolo_weight)

    m = Darknet(cfgfile)
    m.eval()
    m.load_weights(weightfile)
    eval_wid = m.width
    eval_hei = m.height

    print("the model's width and height is :", m.width, "and ", m.height)
    use_cuda = 1
    if use_cuda:
        m.cuda()

    conf_thresh = 0.4
    nms_thresh = 0.4
    iou_thresh = 0.5
    min_box_scale = 8. / m.width
    print("min box scale is ", min_box_scale)

    with open(imglist) as fp:
        lines = fp.readlines()

    total = 0.0
    proposals = 0.0
    correct = 0.0
    lineId = 0
    avg_iou = 0.0
    for line in lines:
        img_path = line.rstrip()
        print("img_path : ", img_path)
        if img_path[0] == '#':  # 取名称的第一个元素
            continue
        lineId = lineId + 1  # 共有多少章图片
        lab_path = img_path.replace('images', 'labels')
        lab_path = lab_path.replace('JPEGImages', 'labels')
        lab_path = lab_path.replace('/crop', '/yolo-labels/crop')  # 这里都替换路径了
        lab_path = lab_path.replace('.jpg', '.txt').replace('.png', '.txt')
        # replace(oldvalue, newvalue)
        print("lab_path : ", str(lab_path))
        truths = read_truths(lab_path)  # 真实标签
        # truths = read_truths_args(lab_path, min_box_scale)
        # truths = read_truths('inria/Train/pos/yolo-labels/crop000607.txt')
        print(truths)
        # imgfile1 = 'inria/Train/pos/crop000607.png'
        # img = Image.open(imgfile1).convert('RGB')

        # img = img.resize((eval_wid, eval_hei))
        img = Image.open(img_path).convert('RGB').resize((eval_wid, eval_hei))
        boxes = do_detect(m, img, conf_thresh, nms_thresh, use_cuda)
        print("length of boxes :",  len(boxes))
        if False:
            savename = "tmp/%06d.jpg" % (lineId)
            print("save %s" % savename)
            plot_boxes(img, boxes, savename)

        total = total + truths.shape[0]  # ground_truth labels的个数
        # 为真实的正例，是TP + FN
        for i in range(len(boxes)):
            if boxes[i][4] > conf_thresh:
                proposals = proposals+1  # 预测出来的检测框数量
                # proposal即为预测为正值的情况，为TP+FP
        for i in range(truths.shape[0]):
            box_gt = [truths[i][1], truths[i][2],
                      truths[i][3], truths[i][4], 1.0]
            best_iou = 0
            for j in range(len(boxes)):
                iou = bbox_iou(box_gt, boxes[j], x1y1x2y2=False)
                best_iou = max(iou, best_iou)  # 最大的IoU
            if best_iou > iou_thresh:
                avg_iou += best_iou
                correct = correct+1  # correct为预测矩阵中的TP，为recall和precision中的分子

    precision = 1.0*correct/proposals  # 对所有图片的结果
    recall = 1.0*correct/total  # 这两个指标怎么定义的？  
    # 简单理解为 recall越低越好
    fscore = 2.0*precision*recall/(precision+recall)
    print("results in recall.py :")
    print("%d IOU: %f, Recal: %f, Precision: %f, Fscore: %f\n" %
          (lineId-1, avg_iou/correct, recall, precision, fscore))


if __name__ == '__main__':
    import sys
    cfgfile = 'cfg/yolov2.cfg'
    weightfile = 'weights/yolov2.weights'
    # imglist = 'inria/Train/pos/yolo-labels/crop_000606.txt'
    imglist = 'test_recall.txt'
    # if len(sys.argv) == 4:
    #     cfgfile = sys.argv[1]
    #     weightfile = sys.argv[2]
    #     imglist = sys.argv[3]
    eval_list(cfgfile, weightfile, imglist)
    # else:
    #     print('Usage:')
    #     print('python recall.py cfgfile weightfile imglist')
        # python recall.py test160.cfg backup/000022.weights face_test.txt
        # 根据示例，imglist应该是个储存图片的txt文件？？
