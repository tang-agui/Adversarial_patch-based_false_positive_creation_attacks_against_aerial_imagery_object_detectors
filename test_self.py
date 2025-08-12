import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms

# a_list = [4,3,2,4]
# a_tensor = torch.tensor(a_list)
# b_list = [5,6,7,8]
# b_tensor = torch.tensor(b_list)
# list_mimus = a_tensor - b_tensor
# print("list minus : ", list_mimus)


# #   用于统计list中特定元素的个数
# a_list = [0,1,1,1,2,3,4,4,4,4,4,5,6]
# print("a_list : ", a_list)
# item_len = a_list.count(1)
# print("item 1 in list : ", item_len)

# a_list = []
# for i in range(15):
#     a_list.append(i)
# print("a_list : ", a_list)
# a = torch.tensor()
# def test_fun(a):
#     '''
#     测试能不能多个return
#     '''
#     for i in range(5):
#         print(" i : ", i)
#         if i > a:
#             return 1
            
              
                  
#     return 3

# print(test_fun(7))

# a = torch.arange(2*2*3).reshape(3,2,2)
# print("a = ", a, " and size of a : ", a.size())
# print("第一列元素： ", a[:,0])
# a_sum = torch.sum(a[0:2,::],dim=0)

# print("after sum : ", a_sum, "and size of a_sum : ", a_sum.size())

# a = torch.arange(3*5).reshape(3,5)
# print("a = ", a, " and size of a : ", a.size())
# # a_sum = torch.sum(a,dim=0)

# # print("after sum : ", a_sum, "and size of a_sum : ", a_sum.size())
# # b = torch.nonzero(a==20)
# # # b = torch.nonzero(a==2).squeeze()
# # print("nonzero b : ", len(b), "size of b : ", b.size())

# a[:,-2:] = 0
# print("a after replace : ", a)

# delta = torch.FloatTensor(size=(2,3)).uniform_(-0.1, 0.1)
# print("delta data : ", delta)


def generate_patch(type,patch_size):
    """
    随机生成一个补丁，用于作为优化的起始点
    Generate a random patch as a starting point for optimization.

    :param type: Can be 'gray' or 'random'. Whether or not generate a gray or a random patch.
    :return:
    """
    if type == 'gray':
        adv_patch_cpu = torch.full(
            (3, patch_size[0], patch_size[1]), 0.5)
    #  使用torch.full等函数生成patch，可以生成灰度和彩色的补丁
    #  返回一个数值全为0.5，维度为前面size参数的向量
    elif type == 'random':
        adv_patch_cpu = torch.rand(
            (3, patch_size[0], patch_size[1]))
        #   torch.rand()生成[0,1]间的均匀分布数据

    return adv_patch_cpu
adv_patch_cpu = generate_patch("random", [80, 80])

im = transforms.ToPILImage('RGB')(adv_patch_cpu)

im.save("testing_trained_patches/random_semantic_patch_tests/random_patch/patch_random.png")

# a = torch.zeros(9)
# print(a)
# #   以下测试torch.nn.CrossEntropy是否需要one-hot格式标签
# a = np.arange(1,13).reshape(3,4)
# b = torch.from_numpy(a)
# input = b.float()
# print('input:\n',input)

# y_target = torch.tensor([0])
# y_target = y_target.repeat(3)
# print('y_target:\n',y_target)

# crossentropyloss=nn.CrossEntropyLoss(reduction='none')
# crossentropyloss_output=crossentropyloss(input,y_target)
# print('crossentropyloss_output:\n',crossentropyloss_output)

# CE_loss = nn.CrossEntropyLoss()
# CE_loss_cal = CE_loss(input, y_target)
# print("crossEntropy loss without reduction : ", CE_loss_cal)

# a = torch.rand(3,4)
# print("data a : ", a)
# b_mean = torch.mean(a, dim=1, keepdim=True)
# mean_b = torch.mean(b_mean)
# print("mean b : ", mean_b)
# print("mean a :", b_mean, '\n', torch.mean(mean_b))
# a_max,_ = torch.max(a, 1, keepdim=True)
# print("max a : ", a_max)
# a_sum = torch.sum(a,1, keepdim=True)
# print("sum of a : ", a_sum)
# test_list = []
# for i in range(3):
#     a = torch.rand(2,3)
#     test_list.append(a)


# print(test_list)
# test_list_cat = torch.cat(test_list, 1)
# print(test_list_cat)
# a = 19
# b = 8
# c_floor = a // b
# c_div = torch.div(a,b,rounding_mode='floor')
# print("floor c : ", c_floor)
# print("c div : ", c_div)
# feature_map = torch.rand(4,3,2)
# feature_size = feature_map.size(-1)
# print("feature size : ", feature_size)



# a_dim = torch.rand(4)
# a_dim = a_dim.view(-1, 1)
# b_dim = torch.rand(4,1)
# print("tensor a : ", a_dim)
# print("tensor b : ", b_dim)

# ab_cat = torch.cat([a_dim, b_dim], 1)
# print("tensor cated : ", ab_cat)
# # a = torch.rand(3,2)
# # rand_loc = torch.rand_like(a)
# print(a)
# a = torch.max(a, torch.tensor(0.2))
# #   torch.max/min除了获得tensor中的最大、最小值外，还可以对值进行比较，从而返回最大或最小值
# #   例如torch.max(a, b)返回相应元素的最大值，即谁大保留谁
# print("a min : ", a)
# a = torch.min(a, torch.tensor(0.8))
# print("tensor after bounded : ", a)







# def lab_transform(lab_batch_origin):
#         '''
#         对原始的lab_batch进行变换，只保留标签数据中size(面积)最大的目标
#         input: lab_batch_origin
#         output: lab_batch_select 
#         '''
#         # lab_batch_select = torch.cuda.FloatTensor(lab_batch_origin.size(0),1,5).fill_(0)
#         lab_batch_select = torch.zeros(lab_batch_origin.size(0),1,5).cuda()
#         area_cal = lab_batch_origin[:,:, 3] * lab_batch_origin[:,:,4]   #   shape = ?
#         print("calculated area : ", area_cal)
#         # area_cal = area_cal.unsqueeze(-1)
#         max_value, max_index = torch.max(area_cal, 1)   #   返回
#         print("max value : ", max_value, '\n', "index : ", max_index)
        
#         for i in range(lab_batch_origin.size(0)):
#             print(lab_batch_select[i,:,:].size())
#             print(lab_batch_origin[i,max_index[i],:])
#             lab_batch_select[i,:,:] = lab_batch_origin[i,max_index[i],:]
#         return lab_batch_select
    
# lab_batch_origin = torch.rand(3,2,5)
# print("original data : ", lab_batch_origin)
# lab_batch = lab_transform(lab_batch_origin) 
# print("data after selected : ", lab_batch)