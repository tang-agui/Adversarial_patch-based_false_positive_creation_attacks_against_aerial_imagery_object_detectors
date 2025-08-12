This project is an official implementation of the paper "Adversarial patch-based false positive creation attacks against aerial imagery object detectors", which has been published in Neurocomputing.
对应本文博士大论文第四章。

与adversarial_patch_attacks_against_optical_object_detectors工程一样，首先对模型权重文件、配置文件及环境进行验证。

- 测试文件：
运行：python clean_img_pre.py，如果运行正常，则说明模型权重、配置已经相关配置正常。
    - imgdir = 待测试/检测的图片，例如："trained_patches_test/physical_test/salient_patches/v2_raw"
    - cfgfile = YOLOv3模型的配置文件，默认："cfg/yolov3-dota.cfg"
    - weightfile = 权重文件，在服务器中，使用全局变量："/mnt/jfs/tangguijian/DOTA_YOLOv3_patch_AT/weights/yolov3-dota_110000.weights"
    - savedir = 检测结果保存文件夹，例如："trained_patches_test/physical_test/salient_patches/v2_raw_pre"  #
其中的imgdir和savedir可根据需要更改，cfgfile和weightfile需要保持相同设置。


- 补丁训练：
    - 训练补丁的主程序。
    - 参数设置：需要重点这是patch_config.py中的包括：
        - self.cfgfile：解析模型架构的参数，默认为"cfg/yolov3-dota.cfg"，即YOLOv3；
        - self.weightfile：YOLOv3的权重文件，默认为/mnt/jfs/tangguijian/DOTA_YOLOv3_patch_AT/weights/yolov3-dota_110000.weights"。
        这里使用的是绝对路径。该文件需要结合具体任务，例如DOTA数据集，VisDrone数据集，训练得到。
        - self.printfile：默认即可。
        - self.patch_size：补丁大小，默认为224
    - 数据集设置：在patch_config.py中的第14-17行（self.img_dir:图片，self.lab_dir：标签格式，本项目使用的标签格式为yolo格式，具体可参考相应数据集下格式要求）
    与adversarial_patch_attacks_against_optical_object_detectors区别在于训练数据设置的不一样。

    - 补丁保存设置：在主程序train_patch.py第372行，设置"save_patch_dir= training_patches_saves/trained_patches"
    补丁保存频率在378行设置。
训练补丁程序耗时较长，建议在后台训练，并保留日志：
nohup python -u train_patch.py > training_patches_saves/training_logs/training_test_log.log 2>&1 &

本工程中设置的情况较多：
- 目标攻击：
    - loss较为复杂，需要根据具体的攻击情况进行设置，具体在train_patch.py主程序的186行后续，包括详细的说明，可参考。
- 非目标攻击
    - loss较为简单，即能使得检测器识别到场景中不存在的目标即可/

更多的参数设置可参考相应的论文。

输出日志中出现：
Running epoch 0:   1%|          | 1/101 [00:25<42:07, 25.28s/it]
Running epoch 0:   2%|▏         | 2/101 [00:26<18:15, 11.06s/it]
表明程序已正常运行。


本project已经在服务器上完成验证，具体使用的是刘旭博士账号下的/home/ubuntu/anaconda3/envs/patchAT虚拟环境。
对应服务器地址为：/mnt/jfs/tangguijian/DOTA_YOLOv3_patch_creation_AT。