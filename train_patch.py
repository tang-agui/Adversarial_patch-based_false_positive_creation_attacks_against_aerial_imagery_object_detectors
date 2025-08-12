"""
Training code for Adversarial patch training
"""
import utils
import PIL
import load_data
import recall_DOTA
from tqdm import tqdm
from torch.backends import cudnn
from load_data import *
import gc  # java中的垃圾回收机制
import matplotlib.pyplot as plt
from torch import autograd
from torchvision import transforms
# from torch.utils.tensorboard import SummaryWriter
import subprocess
import copy
import torch
import patch_config
import sys
import time
import utils_self


TV_FACTOR = 2.5
NPS_FACTOR = 0.01
# TARGET_ID = "Untargeted Attack !"   #   定义该cell中目标类别ID
TARGET_ID = 14  # 定义该cell中目标类别ID
'''
plane
baseball-diamond
bridge
ground-track-field
small-vehicle
large-vehicle
ship
tennis-court
basketball-court
storage-tank
soccer-ball-field
roundabout
harbor
swimming-pool
helicopter
'''


class PatchTrainer(object):  # 


    def __init__(self, mode):
        '''params configuration'''
        self.config = patch_config.patch_configs[mode]()
        print("training mode : ", mode)  # 打印出训练模式
        # print("using sigmoid = True!")
        # self.darknet_model = darknet_self(self.config.cfgfile)
        self.darknet_model = Darknet(self.config.cfgfile)
        self.darknet_model.load_darknet_weights(
            self.config.weightfile)  # 配置架构和权重文件
        self.darknet_model_1 = self.darknet_model.eval().cuda()  # TODO: Why eval?
        # self.darknet_model = self.darknet_model.eval().cuda()  # TODO: Why eval?

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.darknet_model = torch.nn.DataParallel(
                self.darknet_model_1).cuda()
            # self.darknet_model = torch.nn.DataParallel(
            #     self.darknet_model_1, device_ids=[0]).eval().cuda()
        else:
            print("Let's use single GPU!")
            self.darknet_model = self.darknet_model.eval().cuda()
        self.patch_applier = PatchApplier().cuda()
        self.patch_transformer = PatchTransformer().cuda()  # 对patch进行变换后再粘贴到batch数据上
        self.prob_extractor = MaxProbExtractor(
            0, 15, self.config).cuda()  # 更改为目标cls_id
        #----------------------------------------------------#
        #   这里需要更改
        #----------------------------------------------------#
        self.nps_calculator = NPSCalculator(
            self.config.printfile, self.config.patch_size).cuda()
        #  传入的文件为non_printability/30value.txt，以其中的数值为base
        self.total_variation = TotalVariation().cuda()
        self.colorful_loss = load_data.HasSusRGB().cuda()

    def train(self):
        """
        Optimize a patch to generate an adversarial example.
        :return: trained patch-revised by TGJ
        """
        n_png_images = len(fnmatch.filter(
            os.listdir(self.config.img_dir), '*.png'))
        n_jpg_images = len(fnmatch.filter(
            os.listdir(self.config.img_dir), '*.jpg'))
        n_images = n_png_images + n_jpg_images  # 应该和n_images_clean一致
        print("Total images in TrainSet : ", n_images)

        n_lab = len(fnmatch.filter(os.listdir(self.config.lab_dir), '*.txt'))

        length_txt_boxes, _ = utils_self.txt_len_read(self.config.lab_dir)
        print("Total instances in TrainSet : ", length_txt_boxes)

        print("Total txt file in TrainSet : ", n_lab)
        img_size = self.darknet_model_1.height

        print("patch size : ", self.config.patch_size)
        # print("batch size : ", self.config.batch_size)      #   batch size

        """
        print("images size in darknet_model is :", img_size)
        输出调用的模型输入尺寸，例如是yolov2的时候，输入大小是608
        作为加载数据集的参数，用来对输入图片进行resize？？
        """
        batch_size = self.config.batch_size  # 被继承
        print("batch_size = ", batch_size)
        max_n_epochs = 401  # n_epochs = 10000
        max_lab = 252  # 这里进行了筛选，可提高效率
        #   假设这里不用原来的数据，那么计算速度应该可以进一步提高

        adv_patch_cpu = self.generate_patch("random")  #
        # adv_patch_cpu = self.read_image("patches_origin/class_detection.png")
        adv_patch_cpu.requires_grad_(True)  # 用于计算真实梯度

        train_loader = torch.utils.data.DataLoader(DotaDataset(self.config.img_dir, self.config.lab_dir, max_lab, img_size,
                                                               shuffle=True),
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=10)  # 测试阶段shuffle=False
        self.epoch_length = len(train_loader)  # 这里返回总数据集有多少个迭代batch
        print(f'One epoch is {len(train_loader)}')

        optimizer = optim.Adam(
            [adv_patch_cpu], lr=self.config.start_learning_rate, amsgrad=True)
        print("original learning rate : ",
              optimizer.param_groups[0]['lr'])  # 打印优化参数
        # 这个地方设置优化的目标是x还是θ，明确了优化的目标就是adv_patch_cpu
        scheduler = self.config.scheduler_factory(optimizer)  # 不需要更新参数，这一行可以不要

        #   定义是否使用sigmoid激活
        sigmoid_mode = False
        print("sigmoid mode : ", sigmoid_mode)
        # precision_record_test, recall_record_test = [], []
        ep_loss_list = []  # 用于存储训练时det_conf的变化，可能用于说明训练过程

        for epoch in range(max_n_epochs):
            # ep_det_loss = 0
            # ep_bbox_loss = 0
            ep_nps_loss = 0
            ep_tv_loss = 0
            ep_loss = 0
            ep_no_obj_loss = 0.  # creation attack中的obj_loss
            ep_no_cls_loss = 0.  # creation attack中的cls_loss
            ep_colorful_loss = 0.
            # ep_det_obj_loss = 0.
            # ep_det_cls_loss = 0.
            et0 = time.time()

            for i_batch, (img_batch, lab_batch) in tqdm(enumerate(train_loader), desc=f'Running epoch {epoch}', total=self.epoch_length):
                with torch.autograd.detect_anomaly():
                    ###############################################################
                    #   2022-03-22记
                    #   图片本来是[0,255]范围内值，然后InriaDataset预处理数据时对数据进行了处理
                    #   已经变成了tensor，此时在[0,1]
                    ###############################################################
                    img_batch = img_batch.cuda()  # [batch, 3, 608, 608]
                    lab_batch = lab_batch.cuda()  # [batch, max_lab, 5]
                    adv_patch = adv_patch_cpu.cuda()

                    '''
                    训练补丁的时候：
                    1. 旋转角度打开，但是后面的随机位置意义不大，不用关注
                    2. 补丁粘贴的位置现在是随机的，即在训练的时候可以不关心补丁的位置，但是测试时需要关心
                    '''
                    adv_batch_t, patch_center = self.patch_transformer(
                        adv_patch, lab_batch, img_size, do_rotate=True, rand_loc=False)

                    '''
                    adv_batch_t: 粘贴补丁后mask 
                    patch_center: 补丁的中心坐标，[batch, 2]，分别表示每张图片对应补丁的中心坐标
                    '''
                    #   调通完后，数据格式应该是[batch, 1, 3, 608, 608]
                    #   现在还需要返回补丁的中心，用于确定补丁所在的特征cell

                    p_img_batch = self.patch_applier(img_batch, adv_batch_t)

                    #   图片在这里已经转换成了tensor了
                    p_img_batch = F.interpolate(
                        p_img_batch, (self.darknet_model_1.height, self.darknet_model_1.width))

                    # print("p_img_batch size : ", p_img_batch.size(0))
                    # for i in range(p_img_batch.size(0)):
                    #     img_after_patched = p_img_batch[i,:,:,:]
                    #     img_after_patched = transforms.ToPILImage('RGB')(img_after_patched)
                    #     img_name = "patched_" + str(i) + ".png"
                    #     img_savedir = "DOTA_creation_test_imgs/patch_rand_test"
                    #     img_after_patched.save(os.path.join(img_savedir, img_name))

                    outputs = self.darknet_model(p_img_batch)
                    '''
                    [batch, 60, 19, 19]
                    [batch, 60, 38, 38]
                    [batch, 60, 76, 76]
                    '''
                    #########################################################################
                    #   如何找到patch在三个尺度上的位置，并索引到该cell？从而要提取其obj_conf??
                    #########################################################################
                    #   根据返回的坐标分别得到三个尺度上的index，及相应cells的obj_conf和cls_conf
                    index_obj_conf, index_cls_conf = self.obj_cls_conf_find(
                        outputs, img_size, patch_center)
                    '''
                    index_obj_conf: list，3 x batch x 3，三个特征图，
                    index_cls_conf: list，3 x batch x 3 x 15
                    '''
                    no_obj_reshape = self.no_obj_reshape(
                        index_obj_conf)  # 要变成[batch, 9]
                    #   提取概率的时候，三个特征图上都有一个cell，每个cell有3个anchors，
                    no_cls_reshape = self.no_cls_reshape(
                        index_cls_conf)  # 数据size=[batch, 9, 15]
                    #   这里需要注意，不能采用模型的原始输出，应使用经过sigmoid激活的部分
                    #   函数传入模型的原始输出，内部对obj_score和cls_socre首先经过sigmoid处理，然后分别提出
                    #   相对应patch-index处的obj_conf和cls_conf，
                    #   因为每个cell存在3个anchors，因此，最终的输出为：
                    #      index_obj_conf = [[batch, 3], [batch, 3],[batch, 3]]
                    #      index_cls_conf = [[batch, 3, 15], [batch, 3, 15], [batch, 3, 15]]
                    ###############################################################################
                    #   以下求loss
                    ###############################################################################
                    #   1. obj_loss
                    # print("no_obj_reshape", no_obj_reshape)
                    # print("no_obj_reshape size : ", no_obj_reshape.size())
                    obj_conf_max, _ = torch.max(
                        no_obj_reshape, 1, keepdim=True)
                    #   只要9个box中有一个大于阈值就可
                    # obj_conf_sum = torch.sum(no_obj_reshape, dim=1, keepdim=True)
                    #   用sum反而会有副作用，因为是多个boxes求和，那么就不能直接使用（1-no_obj_loss）
                    #   [batch, 1]，求得每张图片中9个anchors最大的obj_conf
                    no_obj_loss = torch.mean(
                        obj_conf_max).cuda()  # 上面两行和下面一行的区别，
                    # 是个最小化问题，使用1/no_obj_loss会得到过大值
                    no_obj_loss = 4*(1 - no_obj_loss)  # 这里娶了mean，那就可以和1取余
                    #   数值量级：1左右
                    #   上面两行先求取每张图片中所有anchors中最大的obj_conf，然后再求平均
                    # no_obj_loss = torch.mean(no_obj_reshape)  #   直接求所有元素的最大值

                    #   2. cls_loss

                    #   目标标签设置为0（plane），此时不需要设置为one-hot形式吗？
                    ##########################################################################
                    # （几点基础补充）
                    # 1. 是的，torch.nn.CrossEntropyLoss()的target不需要one-hot形式
                    # 2. torch.nn.CrossEntropyloss()直接算出了batch的mean loss，即直接得到的是个scalar，不需要再进行处理
                    ##########################################################################
                    #   (1).cls_loss-1(CE-loss)
                    no_cls_loss = self.noCLS_Loss_CE(no_cls_reshape, TARGET_ID).cuda()
                    #   数值量级：2~3
                    # max_det_prob, max_cls_conf = self.prob_extractor(outputs, sigmoid_mode)
                    #   这里，其实也会提取当前场景下指定id的det_conf和cls_conf，但是因为设计的loss发现cls_conf没用
                    #   而对于creation attack，是不是可以用上这一部分？？(试验表明，并没用)
                    #   但此时需要场景中的类别和待攻击类别不一样。
                    #   这里的max_cls_conf是什么维度？[]

                    # # #   (2). cls_loss-2(max_targeted loss)
                    # no_cls_loss = self.noCLS_loss_targeted(no_cls_reshape, TARGET_ID)
                    # #   什么量级？[0~1]

                    # no_cls_loss = (1 - no_cls_loss).cuda()   #   这是直接最大化指定TARGET_ID的概率
                    # #   此时仍然要取余，因为得到的是目标ID的cls_prob，而现在优化范式是minimum。

                    # #   (3). cls_loss-3(max_cls - target_cls)
                    # # ###########################################################################
                    # #   即使使用cls_loss，也可以进一步再细分为上面的（2）、（3）两种情况
                    # ###########################################################################
                    # no_cls_loss = self.noCLS_loss_targeted(no_cls_reshape, TARGET_ID).cuda()
                    # no_cls_loss = 4* no_cls_loss

                    # 给概率提取函数传参batch，是个一维tensor
                    #   同时min之前能检测到的obj_loss，因此这里似乎不影响
                    # 这三个score有区别，一个是yolo分类score，另两个直接和patch相关
                    # 必须直接和patch相关才能使用反向传播更新patch

                    nps = self.nps_calculator(adv_patch)  # tensor  #   <0.1
                    tv = self.total_variation(adv_patch)  # tensor  #   0~1

                    nps_loss = nps*NPS_FACTOR
                    tv_loss = tv * TV_FACTOR

                    # 这里虽然是mean，但是根据链式求导法则，最终都能映射到具体图片中去
                    # det_loss = torch.mean(max_prob)
                    # cls_conf_loss = torch.mean(max_cls_conf)
                    #   这里的cls_conf_loss是取mean，然后minimum，还是也可以使用CE-loss？
                    #   (1) loss-1
                    # loss = det_loss + nps_loss + \
                    #     torch.max(tv_loss, torch.tensor(0.1).cuda()) + \
                    #     no_obj_loss+no_cls_loss
                    #   (2) loss-2（CE-los，目前来看，loss-2效果最好，也最简单）
                    # loss = nps_loss + \
                    #     torch.max(tv_loss, torch.tensor(0.1).cuda()) + \
                    #     no_obj_loss+no_cls_loss
                    #   (3) loss-3, 因为无法同时minimum det_loss和no_obj_loss，但是可以尝试同时minimum cls_conf_loss
                    #   前提是场景中包含的目标和待攻击优化目标类别不一样
                    # loss = cls_conf_loss + nps_loss + \
                    #     torch.max(tv_loss, torch.tensor(0.1).cuda()) + \
                    #     no_obj_loss+no_cls_loss

                    #   (2) loss-4，untargeted_attack，即非目标攻击，此时仅将obj_conf设置为loss
                    # loss = nps_loss + \
                    #     torch.max(tv_loss, torch.tensor(0.1).cuda()) + \
                    #     no_obj_loss

                    #   （5）loss-5，增加补丁显著性（颜色）损失，参考：https://arxiv.org/pdf/1908.08505.pdf
                    
                    colorful_loss = self.colorful_loss(adv_patch)
                    loss = nps_loss + \
                        torch.max(tv_loss, torch.tensor(0.1).cuda()
                                  ) + no_obj_loss + colorful_loss + no_cls_loss
                    # ep_det_loss += det_loss.detach().cpu().numpy()
                    # ep_bbox_loss += bbox_loss.detach().cpu().numpy()
                    ep_nps_loss += nps_loss.detach().cpu().numpy()
                    ep_tv_loss += tv_loss.detach().cpu().numpy()
                    
                    ep_no_obj_loss += no_obj_loss.detach().cpu().numpy()
                    ep_no_cls_loss += no_cls_loss.detach().cpu().numpy()
                    
                    ep_colorful_loss += colorful_loss.detach().cpu().numpy()
                    
                    ep_loss += loss  # 更新每部分的loss

                    loss.backward()  # BP更新梯度
                    optimizer.step()  # 上下两行同时出现使用，更新adv_patch
                    optimizer.zero_grad()
                    adv_patch_cpu.data.clamp_(0, 1)

            scheduler.step(ep_loss)

            et1 = time.time()
            #   计算整个训练集上的平均loss
            # ep_det_loss = ep_det_loss/len(train_loader)
            ep_nps_loss = ep_nps_loss/len(train_loader)
            ep_tv_loss = ep_tv_loss/len(train_loader)
            ep_loss = ep_loss/len(train_loader)
            ep_no_obj_loss = ep_no_obj_loss / len(train_loader)
            ep_no_cls_loss = ep_no_cls_loss / len(train_loader)
            ep_colorful_loss = ep_colorful_loss / len(train_loader)
            # 前面优化时乘上权重因子4，这里为了展示直观，再还原回去
            ep_loss_list.append(ep_no_obj_loss / 4)

            if True:
                print('/n')
                print('  EPOCH NR: ', epoch),
                print('EPOCH LOSS: ', ep_loss)
                # print('  DET LOSS: ', ep_det_loss)

                print('  NPS LOSS: ', ep_nps_loss)
                print('   TV LOSS: ', ep_tv_loss)
                #   2022-10-21：增加creation attack的loss输出
                print('  NO_OBJ LOSS: ', ep_no_obj_loss)
                print('  NO_CLS LOSS: ', ep_no_cls_loss)
                print('  COLORFUL LOSS: ', ep_colorful_loss)
                print('EPOCH TIME: ', et1-et0)

                #im = transforms.ToPILImage('RGB')(adv_patch_cpu)
                # plt.imshow(im)
                # plt.show()
                # im.save("saved_patches/patchnew1.jpg")

            et0 = time.time()

            if epoch % 20 == 0:  # 测试 % 1 == 0 # 每50个epoch保存
                # test
                save_patch = transforms.ToPILImage('RGB')(
                    adv_patch_cpu.cpu())  # debug时每个epoch都保存图片
                # save_patch_dir = "KITTI_DataSet_mix_S/saved_patches_down_Trans_300"
                save_patch_dir = "training_patches_saves/trained_patches"
                # save_patch_dir = "training_saves_colorful/targeted_attack/ID_14"
                print("saved patch dir : ", save_patch_dir)
                save_patch_name = str(epoch) + "_patch.png"
                save_patch.save(os.path.join(save_patch_dir, save_patch_name))

                if epoch > 0:
                    save_patch_name_0 = str(epoch-20) + "_patch.png"
                    save_patch_name_1 = str(epoch) + "_patch.png"
                    patchfile_0 = os.path.join(
                        save_patch_dir, save_patch_name_0)
                    patchfile_1 = os.path.join(
                        save_patch_dir, save_patch_name_1)
                    patch_MSE = utils_self.patch_MSE_calsulator(
                        patchfile_0, patchfile_1)
                    print("MSE-loss between adjacent patch : ", patch_MSE)

        return adv_patch_cpu, ep_loss_list

    def generate_patch(self, type):
        """
        随机生成一个补丁，用于作为优化的起始点
        Generate a random patch as a starting point for optimization.

        :param type: Can be 'gray' or 'random'. Whether or not generate a gray or a random patch.
        :return:
        """
        if type == 'gray':
            adv_patch_cpu = torch.full(
                (3, self.config.patch_size, self.config.patch_size), 0.5)
        #  使用torch.full等函数生成patch，可以生成灰度和彩色的补丁
        #  返回一个数值全为0.5，维度为前面size参数的向量
        elif type == 'random':
            adv_patch_cpu = torch.rand(
                (3, self.config.patch_size, self.config.patch_size))
            #   torch.rand()生成[0,1]间的均匀分布数据

        return adv_patch_cpu

    def read_image(self, path):
        """
        读取一个已经训练好的补丁
        Read an input image to be used as a patch

        :param path: Path to the image to be read.
        :return: Returns the transformed patch as a pytorch Tensor.
        """
        patch_img = Image.open(path).convert('RGB')
        tf = transforms.Resize(
            (self.config.patch_size, self.config.patch_size))
        patch_img = tf(patch_img)
        tf = transforms.ToTensor()

        adv_patch_cpu = tf(patch_img)
        return adv_patch_cpu

    def obj_cls_conf_find(self, outputs, img_size, patch_center):
        '''
        outputs: [batch, 60, 19, 19]
                 [batch, 60, 38, 38]
                 [batch, 60, 76, 76]
        '''
        obj_conf_all = []
        cls_conf_all = []

        for output in outputs:

            obj_conf_inner = []  # 定义空list，用来存放内层（一个）特征图上求得的index，
            cls_conf_inner = []
            batch = output.size(0)
            h = output.size(2)
            w = output.size(3)
            # output_objectness = torch.sigmoid(output[:, 4, :])
            feature_size = output.size(-1)  # 得到当前特征图的尺寸大小, 19 x 19
            feature_scale = img_size / feature_size  # 得到下采样缩放因子(这里是个整数)
            # patch_feature_axis = patch_center // feature_scale      #   得到补丁在特征图上的位置
            #   这里程序出现warning，可能是python语法变了
            patch_feature_axis = torch.div(
                patch_center, feature_scale, rounding_mode='floor')  # [batch, 2]

            # patch_feature_axis = patch_feature_axis + 1     #   上面得到取整的index，这里坐标分别+1，得到此时的坐标索引
            # print("patch coordinates in single feature map : ", patch_feature_axis)
            #   patch_center = [batch, 2]
            #   patch_feature_axis = [batch, 2]
            # index_single_feature = (patch_feature_axis[:,0] * patch_feature_axis[:,1])  #.view(-1,1)   #   [batch, 1]
            #   以上得到了patch在每个图片上的cell-id
            #   以下提取相应index的obj_conf
            output = output.view(batch, 3, 5+15, h * w)
            #
            for i in range(batch):
                # print("index : ", index_single_feature[i])
                index_x = int(patch_feature_axis[i, 0])
                index_y = int(patch_feature_axis[i, 1])
                # index = int(index_x * feature_scale + index_y)   #   索引号应为int型（第（index_x+1）行的第（index_y+1）个元素）
                # 这里乘上的应该是特征图上的大小，不是缩放因子
                index = int(index_x * feature_size + index_y)
                # print("index int : ", index)
                # output_obj_conf_per = output[i, :, 4, index]    #
                output_cells = output[i, :, 4:20, index]  # 后面的概率类别也需要，因此先留下
                #   [x,y,w,h,obj_conf,P(15x1)]
                #   筛选完后为[3, 16(1+15)]
                # 使用sigmoid（同步）进行概率激活，保留obj_conf和cls_conf
                output_cells_sigmoid = torch.sigmoid(output_cells)
                # 提取得到obj_conf
                output_obj_conf = output_cells_sigmoid[:, 0].view(-1, 3)

                # output_obj_conf = output_objectness.view(-1,3)
                obj_conf_inner.append(output_obj_conf)

                output_cls_conf = output_cells_sigmoid[:, 1:16]
                # output_cls_conf = output_cls_conf.transpose(0,1)
                cls_conf_inner.append(output_cls_conf)
            obj_conf_all.append(obj_conf_inner)
            cls_conf_all.append(cls_conf_inner)
        return obj_conf_all, cls_conf_all

    def no_obj_reshape(self, index_obj_conf):
        #   index_obj_conf：[3, batch, 3]
        # no_ObjLoss = []
        batch_size = len(index_obj_conf[0])  # 这里没问题，因为index_obj_conf是个list，
        obj_conf_concate = torch.zeros(3, batch_size, 3)
        for i, obj_conf in enumerate(index_obj_conf):
            obj_conf_cat = torch.cat(obj_conf, 0)  # 对内部的list进行cat操作。使之成为tensor
            obj_conf_concate[i, :, :] = obj_conf_cat
        obj_conf_trans = obj_conf_concate.transpose(0, 1)  # [batch, 3, 3]
        # print("obj_conf before transpose : ", obj_conf_trans)
        # obj_conf_trans = obj_conf_trans.view(batch_size, 9)    #   这里为什么不能用view()
        obj_conf_trans = obj_conf_trans.reshape(batch_size, 9)
        # print("obj_conf_trans after transpose : ", obj_conf_trans)
        #   这样是可行的，对数据格式进行了调整，使之成为batch x N_boxes格式

        return obj_conf_trans

    def no_cls_reshape(self, index_cls_conf):
        #   index_cls_conf: 3 x batch x 3 x 15
        #   对于cls_loss，需要指定攻击类别，可能还要使用CE_loss
        batch_size = len(index_cls_conf[0])  # 得到任意一个尺度元素，然后第一维记为batch_size
        cls_conf_tensor = torch.zeros(
            3, batch_size, 3, 15)  # 对于最外层先定义一个空的tensor用于存放数据
        #
        for i, cls_conf in enumerate(index_cls_conf):
            cls_batch_inner_tensor = torch.zeros(batch_size, 3, 15)
            #   内层也是多个batch list，因此再定义局部list用于存放数据
            # cls_conf_cat = torch.cat(cls_conf, 1)
            #   cls: batch x 3 x 15

            for j, cls_batch in enumerate(cls_conf):
                cls_batch_inner_tensor[j, :, :] = cls_batch  # 内存数据对batch循环并赋值
            cls_conf_tensor[i, :, :, :] = cls_batch_inner_tensor  # 外层tensor赋值
        cls_conf_tensor = cls_conf_tensor.transpose(0, 1)
        cls_conf_reshape = cls_conf_tensor.reshape(batch_size, 9, 15)
        #   使用CE的话需要构造One-hot向量（也不需要，提供的CE算法可处理飞one-hot数据）
        return cls_conf_reshape

    def noCLS_Loss_CE(self, no_cls_reshape, cls_ID):
        '''
        no_cls_reshape: [batch, 9, 15]
        包含batch，然后每张图片又包含9个boxes
        '''
        batch_size = no_cls_reshape.size(0)  # batch_size
        num_anchors = no_cls_reshape.size(1)  # 9 = 3x3

        CE_loss = torch.nn.CrossEntropyLoss()  # 此时会对单张图片9个anchors进行求和平均
        # CE_loss = torch.nn.CrossEntropyLoss(reduction='none')     这里不会对9个anchors进行求和
        target_label = torch.tensor([cls_ID])  # 定义目标标签为指定的TARGET_ID
        target_label = target_label.repeat(num_anchors)
        batch_loss = torch.zeros(batch_size)
        for i in range(batch_size):
            #   no_cls_reshape[i, :, :] = [9, 15]
            # print("per size : ", no_cls_reshape[i, :, :].size())
            ce_loss_per_image = CE_loss(
                no_cls_reshape[i, :, :], target_label)  # 这里相当于batch=9，
            batch_loss[i] = ce_loss_per_image
        #   cls_batch_loss: [batch]
        cls_batch_loss = torch.mean(batch_loss)  # 这里求mean，有没有其它的方式？
        # cls_batch_loss = torch.sum(batch_loss)
        return cls_batch_loss

    def noCLS_loss_targeted(self, no_cls_reshape, cls_ID):
        '''
        使用另一种noCLS_loss，期望能首先targeted attack
        no_cls_reshape: [batch, 9, 15]
        '''
        batch_size = no_cls_reshape.size(0)
        # num_anchors = no_cls_reshape.size(1)
        batch_loss = torch.zeros(batch_size)
        #   case-1
        # for i in range(batch_size):
        #     targeted_prob_per_images = no_cls_reshape[i, :, cls_ID]     #   9x1
        #     batch_loss[i] = torch.mean(targeted_prob_per_images)
        # cls_batch_loss = torch.mean(batch_loss)

        #   case-2
        for i in range(batch_size):
            targeted_prob_per_image = no_cls_reshape[i, :, cls_ID]
            max_prob_per_images, max_index = torch.max(
                no_cls_reshape[i, :, :], dim=1)  # 9x1

            cls_prob_diff = max_prob_per_images - targeted_prob_per_image
            batch_loss[i] = torch.mean(cls_prob_diff)
            # for j in range(targeted_prob_per_image.size(0)):
            #     if (max_index[j] == cls_ID):
        # cls_batch_loss = torch.mean(batch_loss)
        cls_batch_loss = torch.sum(batch_loss)

        return cls_batch_loss


def main():
    """
    # debug
    # """
    # if len(sys.argv) != 2:
    #     print('You need to supply (only) a configuration mode.')
    #     print('Possible modes are:')
    #     print(patch_config.patch_configs)

    # trainer = PatchTrainer(sys.argv[1])  # sys.argv[1]
    """
    print(sys.argv)
    # ['train_patch.py', 'paper_obj']
    把参数打印出来后，所以最终传入的参数只是第二个
    sys.argv得到的是python后的所有参数？？
    """
    # sys.argv = "obj_cls"
    sys.argv = "paper_obj"
    # sys.argv = "exp4_class_only"
    trainer = PatchTrainer(sys.argv)  # 使用非CLI实现debug
    # 这里的参数也不能随便传递，必须和patch_config.py文件中给出的patch_configs默认参数一致
    final_patch, ep_no_obj_loss = trainer.train()
    # np.save("training_saves_stage2/untargeted_attack/ep_no_obj_loss.npy", ep_no_obj_loss)
    # save_patch = transforms.ToPILImage('RGB')(final_patch)
    # save_patch_dir = "KITTI_DataSet_mix/saved_patches"
    # save_patch_name = "final_patch_car_truck.png"
    # save_patch.save(os.path.join(save_patch_dir, save_patch_name))


if __name__ == '__main__':

    print("start training time : ", time.strftime('%Y-%m-%d %H:%M:%S'))
    
    device = torch.device('cuda:0')  # 显卡序号信息
    GPU_device = torch.cuda.get_device_properties(device)  # 显卡信息
    #   输出显卡参数
    print("GPUs : ", GPU_device)

    print("scale factor : ", load_data.SCALE_FACTOR)
    print("TV FACTOR : ", TV_FACTOR)
    print("NPS FACTOR : ", NPS_FACTOR)

    #   分别打印攻击目标ID和目标类别
    print("TARGET ID : ", TARGET_ID)
    class_names = utils.load_class_names('data/dota.names')
    #   这里假设是目标攻击，给出ID的情况则可以
    print("targeted class : ", class_names[TARGET_ID])

    t_begin = time.time()
    main()
    t_end = time.time()
    print('Total training time: {:.4f} minutes'.format(
        (t_end - t_begin) / 60))  # 输出训练时间
