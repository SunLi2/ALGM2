# # import os
# # import shutil
# #
# # # 源文件夹和目标文件夹路径
# # source_folder = "/home/Storage/sll/MattingCode/DataSet/Transparent-460/Black/Training/val/"
# # destination_folder = "/home/Storage/sll/MattingCode/DataSet/Transparent-460/Black/Training/val_copy/"
# #
# # # 确保目标文件夹存在，如果不存在则创建它
# # if not os.path.exists(destination_folder):
# #     os.makedirs(destination_folder)
# #
# # # 遍历源文件夹中的图像文件
# # for filename in os.listdir(source_folder):
# #     if filename.endswith((".jpg", ".png", ".jpeg", ".gif", ".bmp")):
# #         source_path = os.path.join(source_folder, filename)
# #
# #         # 复制图像并重命名为原名加 '_0' 到 '_19'
# #         for i in range(20):
# #             new_filename = f"{os.path.splitext(filename)[0]}_{i}{os.path.splitext(filename)[1]}"
# #             destination_path = os.path.join(destination_folder, new_filename)
# #             shutil.copyfile(source_path, destination_path)
# #
# # print("图像复制完成")
# #
# # -*- coding: utf-8 -*-
# # create time:2022/9/28
# # author:Pengze Li
# import torch
# import torch.nn as nn
# from torch.autograd import Variable
# from torchvision.models import vgg16
# import warnings
#
# warnings.filterwarnings('ignore')
#
#
# # 计算特征提取模块的感知损失
# def vgg16_loss(feature_module,loss_func,y,y_):
#     out=feature_module(y)
#     out_=feature_module(y_)
#     loss=loss_func(out,out_)
#     return loss
#
#
# # 获取指定的特征提取模块
# def get_feature_module(layer_index,device=None):
#     vgg = vgg16(pretrained=True, progress=True).features
#     vgg.eval()
#
#     # 冻结参数
#     for parm in vgg.parameters():
#         parm.requires_grad = False
#
#     feature_module = vgg[0:layer_index + 1]
#     feature_module.to(device)
#     return feature_module
#
#
# # 计算指定的组合模块的感知损失
# class PerceptualLoss(nn.Module):
#     def __init__(self,loss_func,layer_indexs=None,device=None):
#         super(PerceptualLoss, self).__init__()
#         self.creation=loss_func
#         self.layer_indexs=layer_indexs
#         self.device=device
#
#     def forward(self,y,y_):
#         loss=0
#         for index in self.layer_indexs:
#             feature_module=get_feature_module(index,self.device)
#             loss+=vgg16_loss(feature_module,self.creation,y,y_)
#         return loss
#
#
# if __name__ == "__main__":
#     import time
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     x = torch.ones((10, 3, 512, 512))
#     y = torch.zeros((10, 3, 512, 512))
#     x,y=x.to(device),y.to(device)
#
#     layer_indexs = [3, 3, 3, 3]
#     # 基础损失函数：确定使用那种方式构成感知损失，比如MSE、MAE
#     loss_func = nn.MSELoss().to(device)
#     # 感知损失
#     creation = PerceptualLoss(loss_func, layer_indexs, device)
#     start=time.time()
#     perceptual_loss=creation(x,y)
#     end=time.time()
#     print(perceptual_loss)
#     print(end-start)
#
#
#
#
# import os
# import cv2
# import toml
# import argparse
# import numpy as np
#
# import torch
# from torch.nn import functional as F
# # from networks.encoders.MatteFormer import MatteFormer
# import utils
# from   utils import CONFIG
# import networks
#
#
# def remove_prefix_state_dict(state_dict, prefix="module", skip_prefix="decoder"):
#     """
#     remove prefix from the key of pretrained state dict for Data-Parallel
#     """
#     new_state_dict = {}
#     first_state_name = list(state_dict.keys())[0]
#
#     if not first_state_name.startswith(prefix):
#         for key, value in state_dict.items():
#             if not key.startswith(skip_prefix):
#                 new_state_dict[key] = state_dict[key].float()
#                 # new_state_dict[key]=new_state_dict[key].replace("encoder.", "")
#     else:
#         for key, value in state_dict.items():
#             new_key = key[len(prefix) + 1:]
#             new_key=new_key.replace("encoder.", "")
#             if not new_key.startswith(skip_prefix):
#                 new_state_dict[new_key] = state_dict[key].float()
#                 # new_state_dict[key] = new_state_dict[key].replace("encoder.", "")
#
#     return new_state_dict
#
#
#
#
# def Create_Matteformer():
#     model = MatteFormer(embed_dim=96,
#                         depths=[2, 2, 6, 2],  # tiny-model
#                         num_heads=[3, 6, 12, 24],
#                         window_size=7,
#                         mlp_ratio=4.0,
#                         qkv_bias=True,
#                         qk_scale=None,
#                         drop_rate=0.0,
#                         attn_drop_rate=0.0,
#                         drop_path_rate=0.3,
#                         patch_norm=True,
#                         use_checkpoint=False
#                         )
#     model.cuda()
#
#     checkpoint = torch.load("/home/ljh/SLL/data/sll/MattingCode/matteFormer/pretrained/best_model.pth")
#     model.load_state_dict(remove_prefix_state_dict(checkpoint['state_dict']), strict=True)
#     return model
# for name, param in model.named_parameters():
#     print(f"Parameter name: {name}, Size: {param.size()}")
#
#
#
# import os
# from PIL import Image
#
# def get_image_size(image_path):
#     """获取图像的尺寸"""
#     with Image.open(image_path) as img:
#         return img.size
#
# def compare_folders(folder1, folder2):
#     """比较两个文件夹中图像的尺寸是否一致"""
#     inconsistent_images = []
#
#     # 遍历文件夹1中的图像
#     for filename in os.listdir(folder1):
#         if filename.endswith(('.jpg', '.jpeg', '.png')):
#             image_path1 = os.path.join(folder1, filename)
#             image_path2 = os.path.join(folder2, filename.replace('.png','.jpg'))
#
#             # 如果文件夹2中不存在同名的图像，则跳过
#             if not os.path.exists(image_path2):
#                 continue
#
#             size1 = get_image_size(image_path1)
#             size2 = get_image_size(image_path2)
#
#             # 比较图像尺寸是否一致
#             if size1 == size2:
#                 inconsistent_images.append(filename)
#
#     return inconsistent_images
#
# # 指定两个文件夹路径
# folder1 = "/home/sll/data/sll/MattingCode/ContextMatte/PATH/Adobe_Deep_Matting_Dataset/Combined_Dataset/Training_set/Other/low_alpha_copy/"
# folder2 = "/home/ljh/SLL/data/sll/MattingCode/ContextMatte/PATH/Adobe_Deep_Matting_Dataset/Combined_Dataset/Training_set/Other/low_fg_copy_black/"
#
# # 比较两个文件夹中图像的尺寸是否一致
# inconsistent_images = compare_folders(folder1, folder2)
#
# # 输出不一致的图像名称
# if inconsistent_images:
#     print("以下图像的尺寸不一致：")
#     for filename in inconsistent_images:
#         print(filename)
# else:
#     print("两个文件夹中所有图像的尺寸一致。")
#
# num_inconsistent_images = len(inconsistent_images)
# print("尺寸不同的图像数量:", num_inconsistent_images)


# import os
# from PIL import Image
#
# def resize_image(image_path, max_size):
#     """等比例缩放图像，使得最长边的尺寸等于max_size"""
#     with Image.open(image_path) as img:
#         width, height = img.size
#         if width >= height:
#             new_width = max_size
#             new_height = int(height * max_size / width)
#         else:
#             new_height = max_size
#             new_width = int(width * max_size / height)
#         resized_img = img.resize((new_width, new_height), Image.ANTIALIAS)
#         return resized_img
#
# def resize_images_in_folder(input_folder, output_folder, max_size):
#     """将文件夹中的图像等比例缩放，并保存到另一个文件夹"""
#     os.makedirs(output_folder, exist_ok=True)
#     for filename in os.listdir(input_folder):
#         if filename.endswith(('.jpg', '.jpeg', '.png')):
#             input_image_path = os.path.join(input_folder, filename)
#             output_image_path = os.path.join(output_folder, filename)
#             resized_img = resize_image(input_image_path, max_size)
#             resized_img.save(output_image_path)
#
# # 指定输入和输出文件夹路径以及最大尺寸
# input_folder = "/home/ljh/SLL/data/sll/MattingCode/ContextMatte/PATH/Adobe_Deep_Matting_Dataset/Combined_Dataset/Training_set/Other2/trimap_t/"
# output_folder = "/home/ljh/SLL/data/sll/MattingCode/ContextMatte/PATH/Adobe_Deep_Matting_Dataset/Combined_Dataset/Training_set/Other2/low_trimap_t/"
# max_size = 1500
#
# # 对输入文件夹中的图像进行等比例缩放，并保存到输出文件夹
# resize_images_in_folder(input_folder, output_folder, max_size)



# import os
# from PIL import Image
# import numpy as np
#
# def composite_images(alpha_folder, image_folder, output_folder):
#     """合成图像"""
#     os.makedirs(output_folder, exist_ok=True)
#
#     # 遍历alpha文件夹中的图像
#     for filename in os.listdir(alpha_folder):
#         if filename.endswith('.jpg'):
#             # 构造对应的图像I路径
#             image_filename = filename.replace('.png', '.jpg')
#             alpha_path = os.path.join(alpha_folder, filename)
#             image_path = os.path.join(image_folder, image_filename)
#             output_path = os.path.join(output_folder, filename)
#
#             # 打开 alpha 和图像I
#             alpha = Image.open(alpha_path)
#             image = Image.open(image_path)
#
#             # 确保 alpha 和图像I 的尺寸一致
#             alpha = alpha.resize(image.size, Image.ANTIALIAS)
#
#             # 将 PIL Image 转换为 numpy 数组
#             alpha_np = np.array(alpha)
#             image_np = np.array(image)
#
#             # 对图像I的每个通道逐像素乘以alpha通道
#             composite_np = (alpha_np[:, :, np.newaxis] / 255.0) * image_np
#
#             # 创建合成图像的 PIL Image 对象
#             composite_img = Image.fromarray(composite_np.astype('uint8'))
#
#             # 保存合成图像到输出文件夹
#             composite_img.save(output_path)
#
# # 指定 alpha 和图像I 文件夹路径以及输出文件夹路径
# alpha_folder = "/home/ljh/SLL/data/sll/MattingCode/ContextMatte/PATH/Adobe_Deep_Matting_Dataset/Combined_Dataset/Training_set/Other/alpha/"
# image_folder = "/home/ljh/SLL/data/sll/MattingCode/ContextMatte/PATH/Adobe_Deep_Matting_Dataset/Combined_Dataset/Training_set/Other/fg/"
# output_folder = "/home/ljh/SLL/data/sll/MattingCode/ContextMatte/PATH/Adobe_Deep_Matting_Dataset/Combined_Dataset/Training_set/Other/fg_black/"
#
# # 合成图像
# composite_images(alpha_folder, image_folder, output_folder)


import os
import shutil

def copy_images(source_folder, dest_folder, num_copies):
    """将文件夹中的图像复制多次，并命名为原名加_0到_19"""
    os.makedirs(dest_folder, exist_ok=True)

    for filename in os.listdir(source_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            # 构造原文件路径
            source_path = os.path.join(source_folder, filename)
            # 获取文件名和扩展名
            name, ext = os.path.splitext(filename)
            # 循环复制文件
            for i in range(num_copies):
                # 构造目标文件名
                dest_name = f"{name}_{i}{ext}"
                # 构造目标文件路径
                dest_path = os.path.join(dest_folder, dest_name)
                # 复制文件
                shutil.copyfile(source_path, dest_path)

# 指定源文件夹路径、目标文件夹路径和要复制的次数
source_folder = "/home/ljh/SLL/data/sll/MattingCode/ContextMatte/PATH/Adobe_Deep_Matting_Dataset/Combined_Dataset/Training_set/Other2/low_fg_t/"
dest_folder = "/home/ljh/SLL/data/sll/MattingCode/ContextMatte/PATH/Adobe_Deep_Matting_Dataset/Combined_Dataset/Training_set/Other2/low_fg_t_copy/"
num_copies = 20

# 执行复制操作
copy_images(source_folder, dest_folder, num_copies)



# import os
# from PIL import Image
#
# def get_image_size(image_path):
#     """获取图像的尺寸"""
#     with Image.open(image_path) as img:
#         return img.size
#
# def compare_image_sizes(folder1, folder2):
#     """比较两个文件夹中图像的尺寸是否一致"""
#     inconsistent_images = []
#
#     # 获取文件夹1中的图像文件路径
#     image_paths1 = [os.path.join(folder1, filename) for filename in os.listdir(folder1) if is_image_file(filename)]
#
#     # 获取文件夹2中的图像文件路径
#     image_paths2 = [os.path.join(folder2, filename) for filename in os.listdir(folder2) if is_image_file(filename)]
#
#     # 将图像文件路径按文件名排序
#     image_paths1.sort()
#     image_paths2.sort()
#
#     # 比较图像尺寸是否一致
#     for path1, path2 in zip(image_paths1, image_paths2):
#         size1 = get_image_size(path1)
#         size2 = get_image_size(path2)
#         if size1 != size2:
#             inconsistent_images.append((os.path.basename(path1), size1, size2))
#
#     return inconsistent_images
#
# def is_image_file(filename):
#     """检查文件是否为图像文件"""
#     return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))
#
# # 指定两个文件夹路径
# folder1 = "/home/ljh/SLL/data/sll/MattingCode/ContextMatte/PATH/Adobe_Deep_Matting_Dataset/Combined_Dataset/Training_set/Other/low_fg/"
# folder2 = "/home/ljh/SLL/data/sll/MattingCode/ContextMatte/PATH/Adobe_Deep_Matting_Dataset/Combined_Dataset/Training_set/Other/low_alpha/"
#
# # 比较两个文件夹中图像的尺寸是否一致
# inconsistent_images = compare_image_sizes(folder1, folder2)
# l=len(inconsistent_images)
# print(l)
# # 输出不一致的图像名称和尺寸
# if inconsistent_images:
#     print("以下图像的尺寸不一致：")
#     for filename, size1, size2 in inconsistent_images:
#         print(f"{filename}: 尺寸为{size1}和{size2}")
# else:
#     print("两个文件夹中所有图像的尺寸一致。")


# import os
# from PIL import Image
# import torch
# import torchvision.transforms as transforms
# import cv2
# import numpy as np
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# # 输入文件夹路径
# a_folder = '/home/ljh/SLL/data/sll/MattingCode/DataSet/AIM-test/merged/'  # 替换为前景文件夹路径
# b_folder ='/home/ljh/SLL/data/sll/MattingCode/DataSet/Transparent-460/Black/test/composition_black/' # 替换为前景文件夹路径
# c_folder='/home/ljh/SLL/data/sll/MattingCode/AAAA/matteformer_fg_alpha/preDIM/460_FBA/240512_133611_se_color/'
# bg_folder =  '/home/ljh/SLL/data/sll/MattingCode/DataSet/train2014/'  # 背景文件夹路径
# alpha_folder = '/home/ljh/SLL/data/sll/MattingCode/DataSet/AIM-test/alpha/'  # Alpha通道文件夹路径
# # alpha_folder = '/home/ljh/SLL/data/sll/MattingCode/AAAA/CFPNet-main/predDIM/240229_091430_alpha/'
# # 输出文件夹路径
# a_out_folder = '/home/ljh/SLL/data/sll/MattingCode/DataSets/composition_1k/AEM/yuantu'  # 保存第一个前景合成图像的文件夹
# b_out_folder = '/home/ljh/SLL/data/sll/MattingCode/DataSets/duibi2/duibi_240120_114459_max/fggt/' # 保存第二个前景合成图像的文件夹
# c_out_folder = '/home/ljh/SLL/data/sll/MattingCode/DataSets/transprent_460_test/240512_133611_se_color' # 保存第二个前景合成图像的文件夹
# # 获取输入文件夹中的文件列表
# a_files = sorted(os.listdir(a_folder))
# b_files = sorted(os.listdir(b_folder))
# c_files = sorted(os.listdir(c_folder))
# bg_files = os.listdir(bg_folder)
# alpha_files = sorted(os.listdir(alpha_folder))
#
# # 确保输出文件夹存在
# os.makedirs(a_out_folder, exist_ok=True)
# os.makedirs(b_out_folder, exist_ok=True)
# os.makedirs(c_out_folder, exist_ok=True)
#
# # 合成函数
# def composite(fg, bg, alpha):
#     alpha=alpha/255
#     alpha = alpha.unsqueeze(0).permute(1,2,0)
#     result=fg*alpha+bg*(1-alpha)
#     result=result.cpu().numpy()
#     result=result.astype(np.uint8)
#     # result = np.clip(result.cpu().numpy(), 0, 255).astype(np.uint8)
#     return result
#
# a=0
# # 合成并保存图像
# for (a_file, b_file,c_file, bg_file, alpha_file) in zip(a_files, b_files,c_files,bg_files, alpha_files):
#     a_image = cv2.imread(os.path.join(a_folder, a_file))
#     b_image = cv2.imread(os.path.join(b_folder, b_file))
#     c_image = cv2.imread(os.path.join(c_folder, c_file))
#     alpha_image = cv2.imread(os.path.join(alpha_folder, alpha_file), cv2.IMREAD_GRAYSCALE)
#
#     # 调整背景图像尺寸以匹配前景图像
#     bg_image = cv2.imread(os.path.join(bg_folder, bg_file))
#     bg_image = cv2.resize(bg_image, (a_image.shape[1], a_image.shape[0]), interpolation=cv2.INTER_CUBIC)
#
#
#     a_image=torch.tensor(a_image)
#     bg_image = torch.tensor(bg_image)
#     b_image = torch.tensor(b_image)
#     alpha_image = torch.tensor(alpha_image)
#     c_image = torch.tensor(c_image)
#     a_image=a_image.cuda()
#     b_image = b_image.cuda()
#     c_image = c_image.cuda()
#     bg_image=bg_image.cuda()
#     alpha_image=alpha_image.cuda()
#
#     # 合成第一个前景
#     a_composite = composite(a_image, bg_image, alpha_image)
#     cv2.imwrite(os.path.join(a_out_folder, a_file), a_composite)
#     # # 合成第二个前景
#     # b_composite = composite(b_image, bg_image, alpha_image)
#     # cv2.imwrite(os.path.join(b_out_folder, b_file),b_composite)
#
#     # c_composite = composite(c_image, bg_image, alpha_image)
#     # cv2.imwrite(os.path.join(c_out_folder, c_file),c_composite)
#
#     print(f'合成并保存: {a_file} 和 {b_file}和{c_file},第{a}个')
#     a+=1

# import os
# import random
# import numpy as np
# import cv2 as cv
#
#
# def gen_trimap(alpha):
#     k_size = random.choice(range(1, 5))
#     iterations = np.random.randint(1, 20)
#     kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (k_size, k_size))
#     dilated = cv.dilate(alpha, kernel, iterations)
#     eroded = cv.erode(alpha, kernel, iterations)
#     trimap = np.zeros(alpha.shape, dtype=np.uint8)
#     trimap.fill(128)
#     trimap[eroded >= 255] = 255
#     trimap[dilated <= 0] = 0
#     return trimap
#
#
# def process_folder(input_folder, output_folder):
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
#
#     for filename in os.listdir(input_folder):
#         if filename.endswith(".png") or filename.endswith(".jpg"):
#             filepath = os.path.join(input_folder, filename)
#             alpha = cv.imread(filepath, cv.IMREAD_GRAYSCALE)
#             if alpha is not None:
#                 trimap = gen_trimap(alpha)
#                 output_path = os.path.join(output_folder, filename)
#                 cv.imwrite(output_path, trimap)
#             else:
#                 print(f"Failed to read image: {filepath}")
#
#
# input_folder = '/home/ljh/SLL/data/sll/MattingCode/ContextMatte/PATH/Adobe_Deep_Matting_Dataset/Combined_Dataset/Training_set/Other2/alpha_t/'
# output_folder = '/home/ljh/SLL/data/sll/MattingCode/ContextMatte/PATH/Adobe_Deep_Matting_Dataset/Combined_Dataset/Training_set/Other2/trimap_t/'
#
# process_folder(input_folder, output_folder)



