from torch import optim
import torch


class BaseConfig(object):
    """
    Default parameters for all config files.
    """

    def __init__(self):
        """
        Set the defaults.
        """
        self.img_dir = "/mnt/jfs/tangguijian/Data_storage/creation_patch_attackSet/trainset/images"
        self.lab_dir = "/mnt/jfs/tangguijian/Data_storage/creation_patch_attackSet/trainset/yolo-labels"  # 为什么要将标签文件从其它地方传进来？
        self.img_dir_test = '/mnt/jfs/tangguijian/Data_storage/creation_patch_attackSet/testset/images'
        self.lab_dir_test = '/mnt/jfs/tangguijian/Data_storage/creation_patch_attackSet/testset/yolo-labels'  # 测试集

        #   ensemble training
        self.cfgfile = "cfg/yolov3-dota.cfg"  # "cfg/yolo.cfg" for origin
        # 似乎这里也可以用yolov2.cfg，这样的区别在于输入图片的大小
        self.weightfile = "/mnt/jfs/tangguijian/DOTA_YOLOv3_patch_AT/weights/yolov3-dota_110000.weights"

        self.printfile = "non_printability/30values.txt"
        self.patch_size = 224  # 

        self.start_learning_rate = 0.03  # self.start_learning_rate = 0.03
        #   这里到底是0.1还是0.03？

        self.patch_name = 'base'

        self.scheduler_factory = lambda x: optim.lr_scheduler.ReduceLROnPlateau(
            x, 'min', patience=50)
        # self.scheduler_factory = lambda x: optim.lr_scheduler.ReduceLROnPlateau(
        #     x, 'min', factor = 0.5, patience=50)
        '''CLASS torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
        # factor=0.1, patience=10, verbose=False, 
        threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)[SOURCE]
        定义的动态调整学习率
        mode:min/max，分别表示优化的指标不再上升或下降时变化
        factor：表示变化因子，默认为0.1，
        patience：多少个epochs不变时改变学习率，默认为10，设置为50
        verbose：是否打印信息
        
        '''
        #  使用匿名函数定义一些参数
        self.max_tv = 0

        self.batch_size = 16  # 这是基类，在下面被继承时会被重写

        self.loss_target = lambda obj, cls: obj * cls  # 默认为两个参数的乘积

        self.target_loc = torch.tensor([0.,0.,0.01,0.01])


class Experiment1(BaseConfig):
    """
    Model that uses a maximum total variation, tv cannot go below this point.
    """

    def __init__(self):
        """
        Change stuff...
        """
        super().__init__()

        self.patch_name = 'Experiment1'
        self.max_tv = 0.165


class Experiment2HighRes(Experiment1):
    """
    Higher res
    对高分辨率进行试验
    """

    def __init__(self):
        """
        Change stuff...
        """
        super().__init__()

        self.max_tv = 0.165
        self.patch_size = 400
        self.patch_name = 'Exp2HighRes'


class Experiment3LowRes(Experiment1):
    """
    Lower res
    对低分辨率进行试验
    """

    def __init__(self):
        """
        Change stuff...
        """
        super().__init__()

        self.max_tv = 0.165
        self.patch_size = 100
        self.patch_name = "Exp3LowRes"


class Experiment4ClassOnly(Experiment1):
    """
    Only minimise class score.
    """

    def __init__(self):
        """
        Change stuff...
        """
        super().__init__()
        
        self.batch_size = 8  # origin = 8  # 6/10/18/24
        self.patch_size = 224  # 放大，设置为500

        # self.patch_name = 'ObjectOnlyPaper'
        self.max_tv = 0.165

        self.patch_name = 'Experiment4ClassOnly'
        self.loss_target = lambda obj, cls: cls


class ObjectAndClass(BaseConfig):
    """
    obj_conf+cls_conf
    """

    def __init__(self):
       
        super().__init__()

        self.batch_size = 12 # origin = 8  # 6/10/18/24
        self.patch_size = 224  # 放大，设置为500

        self.patch_name = 'ObjectAndClass'
        self.max_tv = 0.165

        self.loss_target = lambda obj, cls: (0.2*obj+0.8*cls)   #   权重参数？
        # self.loss_target = lambda obj, cls: (obj+0.001*cls)
        # self.loss_target = lambda obj, cls: (obj+0.5*cls)


class ReproducePaperObj(BaseConfig):  # 都继承了BaseConfig类
    """
    Reproduce the results from the paper: Generate a patch that minimises object score.
    """

    def __init__(self):
        super().__init__()

        self.batch_size = 24 # origin = 8  # 6/10/18/24
        self.patch_size = 224  # 放大，设置为500

        self.patch_name = 'ObjectOnlyPaper'
        self.max_tv = 0.165

        # self.loss_target = lambda obj, cls: obj+cls
        self.loss_target = lambda obj, cls: obj
        # lambda函数的语法，冒号前是参数，后是返回值
        # 对于这种情况，只返回目标检测的概率，对应`load_data.py`中的86行


patch_configs = {
    "base": BaseConfig,
    "exp1": Experiment1,
    "obj_cls": ObjectAndClass,
    "exp2_high_res": Experiment2HighRes,
    "exp3_low_res": Experiment3LowRes,
    "exp4_class_only": Experiment4ClassOnly,
    "paper_obj": ReproducePaperObj  # 再生成和文中最小化目标得分的补丁
}
# 定义的参数是个字典，通过键访问值