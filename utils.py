import sys
import os
import time
import math
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torch.autograd import Variable
import torchvision


import struct  # get_image_size
import imghdr  # get_image_size
ANCHOR_PATH = "data/yolov3_anchors.txt"


def sigmoid(x):
    return 1.0/(math.exp(-x)+1.)


def softmax(x):
    x = torch.exp(x - torch.max(x))
    x = x/x.sum()
    return x


def bbox_iou(box1, box2, x1y1x2y2=True):
    if x1y1x2y2:
        mx = min(box1[0], box2[0])
        Mx = max(box1[2], box2[2])
        my = min(box1[1], box2[1])
        My = max(box1[3], box2[3])
        w1 = box1[2] - box1[0]
        h1 = box1[3] - box1[1]
        w2 = box2[2] - box2[0]
        h2 = box2[3] - box2[1]
    else:
        mx = min(box1[0]-box1[2]/2.0, box2[0]-box2[2]/2.0)
        Mx = max(box1[0]+box1[2]/2.0, box2[0]+box2[2]/2.0)
        my = min(box1[1]-box1[3]/2.0, box2[1]-box2[3]/2.0)
        My = max(box1[1]+box1[3]/2.0, box2[1]+box2[3]/2.0)
        w1 = box1[2]
        h1 = box1[3]
        w2 = box2[2]
        h2 = box2[3]
    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
    carea = 0
    if cw <= 0 or ch <= 0:
        return 0.0

    area1 = w1 * h1
    area2 = w2 * h2
    carea = cw * ch  # 可结合图形理解，此为交面积
    uarea = area1 + area2 - carea
    return carea/uarea  # 返回两个盒子的交并比


def bbox_ious(boxes1, boxes2, x1y1x2y2=True):
    if x1y1x2y2:
        mx = torch.min(boxes1[0], boxes2[0])
        Mx = torch.max(boxes1[2], boxes2[2])
        my = torch.min(boxes1[1], boxes2[1])
        My = torch.max(boxes1[3], boxes2[3])
        w1 = boxes1[2] - boxes1[0]
        h1 = boxes1[3] - boxes1[1]
        w2 = boxes2[2] - boxes2[0]
        h2 = boxes2[3] - boxes2[1]
    else:
        mx = torch.min(boxes1[0]-boxes1[2]/2.0, boxes2[0]-boxes2[2]/2.0)
        Mx = torch.max(boxes1[0]+boxes1[2]/2.0, boxes2[0]+boxes2[2]/2.0)
        my = torch.min(boxes1[1]-boxes1[3]/2.0, boxes2[1]-boxes2[3]/2.0)
        My = torch.max(boxes1[1]+boxes1[3]/2.0, boxes2[1]+boxes2[3]/2.0)
        w1 = boxes1[2]
        h1 = boxes1[3]
        w2 = boxes2[2]
        h2 = boxes2[3]
    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
    mask = ((cw <= 0) + (ch <= 0) > 0)
    area1 = w1 * h1
    area2 = w2 * h2
    carea = cw * ch
    carea[mask] = 0
    uarea = area1 + area2 - carea
    return carea/uarea


def nms(boxes, nms_thresh):
    if len(boxes) == 0:
        return boxes  # 如果没有一个框

    det_confs = torch.zeros(len(boxes))
    for i in range(len(boxes)):
        det_confs[i] = 1-boxes[i][4]  # 第5个元素是置信度

    _, sortIds = torch.sort(det_confs)
    out_boxes = []
    for i in range(len(boxes)):
        box_i = boxes[sortIds[i]]
        if box_i[4] > 0:
            out_boxes.append(box_i)
            for j in range(i+1, len(boxes)):
                box_j = boxes[sortIds[j]]
                if bbox_iou(box_i, box_j, x1y1x2y2=False) > nms_thresh:
                    #print(box_i, box_j, bbox_iou(box_i, box_j, x1y1x2y2=False))
                    box_j[4] = 0  # 如果交并比大于阈值，则赋值为0
    return out_boxes  # 返回的仍是boxes信息，筛选后


def convert2cpu(gpu_matrix):
    return torch.FloatTensor(gpu_matrix.size()).copy_(gpu_matrix)


def convert2cpu_long(gpu_matrix):
    return torch.LongTensor(gpu_matrix.size()).copy_(gpu_matrix)

# def get_region_boxes(output, conf_thresh, num_classes, anchors, num_anchors, only_objectness=1, validation=False):


def get_region_boxes(output, conf_thresh, num_classes, anchors, num_anchors, img_size, only_objectness=0, validation=False):
    # anchor_step = len(anchors)//num_anchors
    '''多维度输入'''
    all_boxes = []
    if output.dim() == 3:
        output = output.unsqueeze(0)
    batch = output.size(0)
    assert(output.size(1) == (5+num_classes)*num_anchors)
    h = output.size(2)
    w = output.size(3)
    stride_h = img_size[1] / h
    stride_w = img_size[0] / w

    scaled_anchors = [(anchor_width / stride_w, anchor_height / stride_h)
                      for anchor_width, anchor_height in anchors]

    t0 = time.time()

    # print(output.size())
    output = output.view(batch*num_anchors, 5+num_classes, h*w)
    # print(output.size())
    output = output.transpose(0, 1).contiguous()
    # print(output.size())
    output = output.view(5+num_classes, batch*num_anchors*h*w)
    # print(output.size())
    grid_x = torch.linspace(0, w-1, w).repeat(h, 1).repeat(batch *
                                                           num_anchors, 1, 1).view(batch*num_anchors*h*w).cuda()
    grid_y = torch.linspace(0, h-1, h).repeat(w, 1).t().repeat(batch *
                                                               num_anchors, 1, 1).view(batch*num_anchors*h*w).cuda()
    xs = torch.sigmoid(output[0]) + grid_x
    ys = torch.sigmoid(output[1]) + grid_y

    anchor_w = torch.Tensor(scaled_anchors).index_select(
        1, torch.LongTensor([0]))
    anchor_h = torch.Tensor(scaled_anchors).index_select(
        1, torch.LongTensor([1]))
    # anchor_w = torch.Tensor(anchors).view(num_anchors, anchor_step).index_select(1, torch.LongTensor([0]))
    # anchor_h = torch.Tensor(anchors).view(num_anchors, anchor_step).index_select(1, torch.LongTensor([1]))
    anchor_w = anchor_w.repeat(batch, 1).repeat(
        1, 1, h*w).view(batch*num_anchors*h*w).cuda()
    anchor_h = anchor_h.repeat(batch, 1).repeat(
        1, 1, h*w).view(batch*num_anchors*h*w).cuda()

    ws = torch.exp(output[2]) * anchor_w
    hs = torch.exp(output[3]) * anchor_h

    xs = xs * stride_w
    ys = ys * stride_h
    ws = ws * stride_w
    hs = hs * stride_h

    '''xs = xs / w
    ys = ys / h
    ws = ws / w
    hs = hs / h'''# 这里可进行归一化处理

    det_confs = torch.sigmoid(output[4])

    # cls_confs = torch.nn.Softmax()(Variable(output[5:5+num_classes].transpose(0,1))).data
    #   这里也有疑问
    # cls_confs_temp = output[5:5+num_classes]
    cls_confs = torch.sigmoid(output[5:].transpose(0, 1)).data
    # output[5:5+num_classes].transpose(0,1)
    # print("size of cls_confs", cls_confs.size())
    cls_max_confs, cls_max_ids = torch.max(cls_confs, 1)
    cls_max_confs = cls_max_confs.view(-1)
    cls_max_ids = cls_max_ids.view(-1)
    t1 = time.time()

    sz_hw = h*w
    sz_hwa = sz_hw*num_anchors
    det_confs = convert2cpu(det_confs)
    cls_max_confs = convert2cpu(cls_max_confs)
    cls_max_ids = convert2cpu_long(cls_max_ids)
    xs = convert2cpu(xs)
    ys = convert2cpu(ys)
    ws = convert2cpu(ws)
    hs = convert2cpu(hs)
    if validation:
        cls_confs = convert2cpu(cls_confs.view(-1, num_classes))
    t2 = time.time()
    for b in range(batch):
        boxes = []
        for cy in range(h):
            for cx in range(w):
                for i in range(num_anchors):
                    ind = b*sz_hwa + i*sz_hw + cy*w + cx
                    det_conf = det_confs[ind]
                    if only_objectness:
                        #   这里只是得到筛选参数，但后面输出的时候仍是两个conf都有输出
                        conf = det_confs[ind]
                    else:
                        conf = det_confs[ind] * cls_max_confs[ind]

                    if conf > conf_thresh:
                        bcx = xs[ind]
                        bcy = ys[ind]
                        bw = ws[ind]
                        bh = hs[ind]
                        cls_max_conf = cls_max_confs[ind]
                        cls_max_id = cls_max_ids[ind]
                        box = [bcx, bcy, bw, bh, det_conf,
                               cls_max_conf, cls_max_id]    # 未归一化
                        # box = [bcx/w, bcy/h, bw/w, bh/h, det_conf, cls_max_conf, cls_max_id]
                        if (not only_objectness) and validation:
                            for c in range(num_classes):
                                tmp_conf = cls_confs[ind][c]
                                if c != cls_max_id and det_confs[ind]*tmp_conf > conf_thresh:
                                    box.append(tmp_conf)
                                    box.append(c)
                        boxes.append(box)
        all_boxes.append(boxes)
        t3 = time.time()
    if False:
        print('---------------------------------')
        print('matrix computation : %f' % (t1-t0))
        print('        gpu to cpu : %f' % (t2-t1))
        print('      boxes filter : %f' % (t3-t2))
        print('---------------------------------')

    return all_boxes  # 返回的是所有的boxes（信息）


def plot_boxes_cv2(img, boxes, savename=None, class_names=None, color=None):
    import cv2
    colors = torch.FloatTensor([[1, 0, 1], [0, 0, 1], [0, 1, 1], [
                               0, 1, 0], [1, 1, 0], [1, 0, 0]])

    def get_color(c, x, max_val):
        ratio = float(x)/max_val * 5
        i = int(math.floor(ratio))
        j = int(math.ceil(ratio))
        ratio = ratio - i
        r = (1-ratio) * colors[i][c] + ratio*colors[j][c]
        return int(r*255)

    width = img.shape[1]
    height = img.shape[0]
    for i in range(len(boxes)):
        box = boxes[i]
        x1 = int(round((box[0] - box[2]/2.0) * width))
        y1 = int(round((box[1] - box[3]/2.0) * height))
        x2 = int(round((box[0] + box[2]/2.0) * width))
        y2 = int(round((box[1] + box[3]/2.0) * height))

        if color:
            rgb = color
        else:
            rgb = (255, 0, 0)
        if len(box) >= 7 and class_names:
            cls_conf = box[5]
            cls_id = box[6]
            print('%s: %f' % (class_names[cls_id], cls_conf))
            classes = len(class_names)
            offset = cls_id * 123457 % classes
            red = get_color(2, offset, classes)
            green = get_color(1, offset, classes)
            blue = get_color(0, offset, classes)
            if color is None:
                rgb = (red, green, blue)
            img = cv2.putText(
                img, class_names[cls_id], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.2, rgb, 1)
        img = cv2.rectangle(img, (x1, y1), (x2, y2), rgb, 1)
    if savename:
        print("save plot results to %s" % savename)
        cv2.imwrite(savename, img)
    return img


def plot_boxes(img, boxes, savename=None, class_names=None):
    colors = torch.FloatTensor([[1, 0, 1], [0, 0, 1], [0, 1, 1], [
                               0, 1, 0], [1, 1, 0], [1, 0, 0]])

    def get_color(c, x, max_val):
        ratio = float(x)/max_val * 5
        i = int(math.floor(ratio))
        j = int(math.ceil(ratio))
        ratio = ratio - i
        r = (1-ratio) * colors[i][c] + ratio*colors[j][c]
        return int(r*255)

    width = img.width
    height = img.height
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font='data/simhei.ttf',
                              size=np.floor(3e-2 * width + 0.5).astype('int32'))

    for box in boxes:
        # box = boxes[i]
        x1 = (box[0] - box[2]/2.0) * width  # box[1,2,3,4]中的值是归一化的，这里再还原
        y1 = (box[1] - box[3]/2.0) * height
        x2 = (box[0] + box[2]/2.0) * width
        y2 = (box[1] + box[3]/2.0) * height
        # x1 = box[0]
        # y1 = box[1]
        # x2 = box[2]
        # y2 = box[3]

        rgb = (255, 0, 0)
        if len(box) >= 5 and class_names:  #
            cls_conf = box[4]
            cls_id = int(box[6])
            # 同时输出类别id，str和置信度
            # print('[%i]%s: %f' % (cls_id, class_names[cls_id], cls_conf))
            # print('[%i]%s: %f' % (cls_id, class_names[cls_id], box[4]*box[5]))

            classes = len(class_names)
            offset = cls_id * 123457 % classes  # offset根据不同的类别设置颜色
            red = get_color(2, offset, classes)
            green = get_color(1, offset, classes)
            blue = get_color(0, offset, classes)
            rgb = (red, green, blue)
            # score = box[4]  # 是使用那个score？
            score = box[4] * box[5]    #   yolov3采用的就是两个乘积
            label = '{}{:.2f}'.format(class_names[cls_id], score)

            # label_size = draw.textsize(label, font)
            # label = label.encode('utf-8')
            # text_origin = np.array([x1, y1])
            
            label_size = draw.textlength(label, font)   #   计算给定文本在给定文字下的宽度
            # draw.text
            label = label.encode('utf-8')
            
            label_h = 'so'  #   现在的draw.textlength似乎没法像draw.textsize()一样，
            #   同时获得宽高数据，因此直接认为引入字符串的高度信息

            # label_size = draw.textsize(label, font)
            label_size_h = draw.textlength(label_h, font)
            label_h = label_h.encode('utf-8')
            
            text_origin = np.array([x1, y1])
            
            text_end = np.array([x1+label_size, y1+label_size_h])

            '''
            # font = ImageFont.truetype(None, size=40, encoding="unic")
            draw.text((x1+40, y1), str(round(box[5].item(), 3)), fill=(0,0,205))  # 只是det_confs，并不是cls_confs
            #   round是显示小数点问题，输出的到底是cls_conf还是cls_conf*obj_conf?
            draw.text((x1, y1), class_names[cls_id], fill=(255,0,0))  # 更改字体颜色
            # draw.text((x1, y1), class_names[cls_id], fill=rgb)  # 标注文字
            '''
            # draw.rectangle([tuple(text_origin), tuple(
            #     text_origin+label_size)], fill=(255, 0, 0))
            draw.rectangle([tuple(text_origin), tuple(
                text_end)], fill=(255, 0, 0))
            draw.text(text_origin, str(label, 'UTF-8'),
                      fill=(0, 0, 0), font=font)
        # draw.rectangle([x1, y1, x2, y2], outline=rgb)
        # 作者考虑到了给labels数据标注边框的情况，此时只标注边框，赞
        draw.rectangle([x1, y1, x2, y2], outline=rgb, width=2)
        # 想更改线条的粗细
    if savename:
        # print("save plot results to %s" % savename)  # 输出保存地址
        img.save(savename)
    return img


def read_truths(lab_path):
    if not os.path.exists(lab_path):
        return np.array([])
    if os.path.getsize(lab_path):
        truths = np.loadtxt(lab_path)
        # to avoid single truth problem
        # truths = truths.reshape(truths.size//6, 6)
        truths = truths.reshape(truths.size//5, 5)  # suv_patch_real_suv.py
        return truths
    else:
        return np.array([])

def read_truths_pre_7(lab_path):
    if not os.path.exists(lab_path):
        return np.array([])
    if os.path.getsize(lab_path):
        truths = np.loadtxt(lab_path)
        # to avoid single truth problem
        # truths = truths.reshape(truths.size//6, 6)
        truths = truths.reshape(truths.size//7, 7)  # suv_patch_real_suv.py
        return truths
    else:
        return np.array([])


def read_truths_args(lab_path, min_box_scale):
    truths = read_truths(lab_path)
    new_truths = []
    # remove truths of which the width is smaller then the min_box_scale
    for i in range(truths.shape[0]):
        if truths[i][3] < min_box_scale:
            continue
        new_truths.append([truths[i][0], truths[i][1],
                           truths[i][2], truths[i][3], truths[i][4]])
    return np.array(new_truths)


def load_class_names(namesfile):  # 导入数据集的类名文件 # 用于加载coco.names文件夹
    class_names = []
    with open(namesfile, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.rstrip()
        class_names.append(line)
    return class_names


def image2torch(img):
    width = img.width
    height = img.height
    img = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
    img = img.view(height, width, 3).transpose(
        0, 1).transpose(0, 2).contiguous()
    img = img.view(1, 3, height, width)
    img = img.float().div(255.0)
    return img


def get_anchors(anchors_path):
    anchors_path = os.path.expanduser(anchors_path)
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]

    return np.array(anchors).reshape([-1, 3, 2])[::-1, :, :]


def do_detect(model, img, conf_thresh, nms_thresh, use_cuda=1):
    """
    根据模型和图片、score阈值、nms阈值实现边界框的预测
    """
    model.eval()
    t0 = time.time()

    if isinstance(img, Image.Image):
        """
        isinstance用来判断一个变量的类型，
        例如上面的条件是判断img是否是图片
        并对图片格式进行转换
        adv_patch = Image.open('data/horse.jpg').convert('RGB')  # 将这张图片作为patch进行测试
        adv_patch = adv_patch.resize((patch_size, patch_size))
        transform = transforms.ToTensor()
        adv_patch = transform(adv_patch).cuda()
        """
        width = img.width
        height = img.height
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
        # trans to tensor，并且这个时候还对通道进行了分离，
        # print("img size is :", img.size())  # 608*608*3
        img = img.view(height, width, 3).transpose(
            0, 1).transpose(0, 2).contiguous()  # 对图片进行升维并变换，先[0, 1]，后[0,2]
        # contiguous：把tensor变成内存中连续分布的形式
        img = img.view(1, 3, height, width)
        img = img.float().div(255.0)  # 归一化
    elif type(img) == np.ndarray:  # cv2 image
        img = torch.from_numpy(img.transpose(
            2, 0, 1)).float().div(255.0).unsqueeze(0)
    else:
        print("unknown image type")
        exit(-1)

    t1 = time.time()

    if use_cuda:
        img = img.cuda()

    # Simen: dit doet een forward, vervangen voor duidelijkheid
    with torch.no_grad():
        outputs = model.forward(img)  # prediction

    # get_region_boxes的输入和do_detect()的输入，变成了

    anchors = get_anchors(ANCHOR_PATH)
    num_anchors = len(anchors)
    class_names = load_class_names('data/dota.names')
    num_classes = len(class_names)
    # print("length of class_names : ", len(class_names))
    boxes_list = []
    for i in range(len(anchors)):
        boxes = get_region_boxes(
            outputs[i], conf_thresh, num_classes, anchors[i], num_anchors, (width, height))[0]
        #   get_region_boxes得到的是归一化的boxes
        #   进行调整，得到的是非归一化的boxes
        boxes_list.append(boxes)

    all_boxes = []
    for box in boxes_list:
        for i in range(len(box)):
            box[i][0] = box[i][0] / width
            box[i][2] = box[i][2] / width
            box[i][1] = box[i][1] / height
            box[i][3] = box[i][3] / height
            all_boxes.append(box[i])
    boxes = nms(all_boxes, nms_thresh)  # 在这个函数内进一步对boxes进行了筛选
    #   nms返回的是个list

    return boxes


def read_data_cfg(datacfg):  # 读取cfg文件
    options = dict()
    options['gpus'] = '0,1,2,3'
    options['num_workers'] = '10'
    with open(datacfg, 'r') as fp:
        lines = fp.readlines()

    for line in lines:
        line = line.strip()
        if line == '':
            continue
        key, value = line.split('=')
        key = key.strip()
        value = value.strip()
        options[key] = value
    return options


def scale_bboxes(bboxes, width, height):
    import copy
    dets = copy.deepcopy(bboxes)  # 深拷贝
    for i in range(len(dets)):
        dets[i][0] = dets[i][0] * width
        dets[i][1] = dets[i][1] * height
        dets[i][2] = dets[i][2] * width
        dets[i][3] = dets[i][3] * height
    return dets


def file_lines(thefilepath):
    count = 0
    thefile = open(thefilepath, 'rb')
    while True:
        buffer = thefile.read(8192*1024)
        if not buffer:
            break
        count += buffer.count('\n')
    thefile.close()
    return count


def get_image_size(fname):
    '''Determine the image type of fhandle and return its size.
    from draco'''
    with open(fname, 'rb') as fhandle:
        head = fhandle.read(24)
        if len(head) != 24:
            return
        if imghdr.what(fname) == 'png':
            check = struct.unpack('>i', head[4:8])[0]
            if check != 0x0d0a1a0a:
                return
            width, height = struct.unpack('>ii', head[16:24])
        elif imghdr.what(fname) == 'gif':
            width, height = struct.unpack('<HH', head[6:10])
        elif imghdr.what(fname) == 'jpeg' or imghdr.what(fname) == 'jpg':
            try:
                fhandle.seek(0)  # Read 0xff next
                size = 2
                ftype = 0
                while not 0xc0 <= ftype <= 0xcf:
                    fhandle.seek(size, 1)
                    byte = fhandle.read(1)
                    while ord(byte) == 0xff:
                        byte = fhandle.read(1)
                    ftype = ord(byte)
                    size = struct.unpack('>H', fhandle.read(2))[0] - 2
                # We are at a SOFn block
                fhandle.seek(1, 1)  # Skip `precision' byte.
                height, width = struct.unpack('>HH', fhandle.read(4))
            except Exception:  # IGNORE:W0703
                return
        else:
            return
        return width, height


def logging(message):
    print('%s %s' % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), message))


def xywh2xyxy(x):
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) -
             torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    # iou = inter / (area1 + area2 - inter)
    return inter / (area1[:, None] + area2 - inter)


def non_max_suppression(prediction, conf_thres=0.5, iou_thres=0.5, classes=None, agnostic=False, labels=()):
    """Performs Non-Maximum Suppression (NMS) on inference results
    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates  # obj_conf第一次筛选

    # Settings
    # (pixels) minimum and maximum box width and height
    max_wh = 4096
    max_det = 300  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 1.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = True  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)
              ] * prediction.shape[0]

    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[
                conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            # sort by confidence
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        # boxes (offset by class), scores
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float(
            ) / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output
