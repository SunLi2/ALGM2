# import numpy as np
# #path to provided foreground images
# fg_path = '/home/ljh/SLL/data/sll/MattingCode/ContextMatte/PATH/Adobe_Deep_Matting_Dataset/Combined_Dataset/Training_set/Other/fg/'
# # fg_path = "/home/ljh/SLL/data/sll/MattingCode/DataSet/Transparent-460/Val/val/"
#
# # # path to provided alpha mattes
# # # a_path = 'mask/'
# a_path = "/home/ljh/SLL/data/sll/MattingCode/ContextMatte/PATH/Adobe_Deep_Matting_Dataset/Combined_Dataset/Training_set/Other/alpha/"
# # # Path to background images (MSCOCO)
# bg_path = '/home/sll/data/sll/MattingCode/DataSet/train2014/'
#
# # # Path to folder where you want the composited images to go
# out_path = '/home/ljh/SLL/data/sll/MattingCode/ContextMatte/PATH/Adobe_Deep_Matting_Dataset/Combined_Dataset/Training_set/Other/composition_bai/'
#
# ##############################################################
#
# from PIL import Image
# import os
# import torch
# import cv2
#
# def composite(fg, bg, alpha):
#     alpha=alpha/255
#     alpha = alpha.unsqueeze(0).permute(1,2,0)
#     result=fg*alpha+bg*(1-alpha)
#     result=result.cpu().numpy()
#     result=result.astype(np.uint8)
#     return result
#
# num_bgs = 20
#
# fg_files = os.listdir(fg_path)
# a_files = os.listdir(a_path)
# bg_files = os.listdir(bg_path)
#
# bg_iter = iter(bg_files)
# for im_name in fg_files:
#
#     im = cv2.imread(fg_path + im_name);
#     a = cv2.imread(a_path + im_name,0);
#     im=torch.tensor(im)
#     a=torch.tensor(a)
#     im=im.cuda()
#     a=a.cuda()
#
#     bcount = 0
#     for i in range(num_bgs):
#         bg_name = next(bg_iter)
#         bg = cv2.imread(bg_path + bg_name)
#         bg_image = cv2.resize(bg, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_CUBIC)
#         bg_image=torch.tensor(bg_image)
#         bg_image=bg_image.cuda()
#         out = composite(im, bg_image, a)
#         cv2.imwrite(os.path.join(out_path, f'{im_name[:len(im_name) - 4]}_{bcount}.png'), out)
#         print(im_name,bcount)
#         bcount += 1


# import os
# from PIL import Image
#
#
# def resize_image(image_path, output_folder, max_size):
#     """
#     Resize an image if its longest side is greater than max_size and save to output folder.
#
#     Parameters:
#     - image_path: str, path to the image file.
#     - output_folder: str, path to the folder where resized images will be saved.
#     - max_size: int, the maximum size for the longest side of the image.
#     """
#     with Image.open(image_path) as img:
#         width, height = img.size
#         longest_side = max(width, height)
#
#         if longest_side > max_size:
#             scale = max_size / float(longest_side)
#             new_size = (int(width * scale), int(height * scale))
#             img = img.resize(new_size, Image.ANTIALIAS)
#
#             # Create output folder if it doesn't exist
#             if not os.path.exists(output_folder):
#                 os.makedirs(output_folder)
#
#             # Create the output file path
#             base_name = os.path.basename(image_path)
#             output_path = os.path.join(output_folder, base_name)
#
#             img.save(output_path)
#             print(f"Resized image saved at: {output_path}, New size: {new_size}")
#         else:
#             print(f"Image {image_path} is smaller than {max_size}, no resize needed.")
#             # Copy the original image to the output folder
#             output_path = os.path.join(output_folder, os.path.basename(image_path))
#             img.save(output_path)
#
#
# def resize_images_in_folder(input_folder, output_folder, max_size=1500):
#     """
#     Resize images in an input folder if their longest side is greater than max_size
#     and save them to an output folder.
#
#     Parameters:
#     - input_folder: str, path to the folder containing images to resize.
#     - output_folder: str, path to the folder where resized images will be saved.
#     - max_size: int, the maximum size for the longest side of the images.
#     """
#     for root, _, files in os.walk(input_folder):
#         for file in files:
#             if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
#                 image_path = os.path.join(root, file)
#                 resize_image(image_path, output_folder, max_size)
#
#
# # Example usage:
# input_folder = '/home/ljh/SLL/data/sll/MattingCode/ContextMatte/PATH/Adobe_Deep_Matting_Dataset/Combined_Dataset/Training_set/Other/fg/'
# output_folder = '/home/ljh/SLL/data/sll/MattingCode/ContextMatte/PATH/Adobe_Deep_Matting_Dataset/Combined_Dataset/Training_set/Other/low_fg/'
# resize_images_in_folder(input_folder, output_folder, max_size=1500)


import os
import shutil

# 文件夹路径
folder_a = '/home/ljh/SLL/data/sll/MattingCode/DataSet/AIM-test/alpha/'
folder_b = '/home/ljh/SLL/data/sll/MattingCode/AEMatter/alphasAIM/'
folder_c = '/home/ljh/SLL/data/sll/MattingCode/DataSet/AIM-test/AEM_alpha/'

# 确保目标文件夹存在
os.makedirs(folder_c, exist_ok=True)

# 获取文件夹a中的所有文件名
files_in_a = set(os.listdir(folder_a))

# 遍历文件夹b，检查是否有与文件夹a同名的文件
for file_name in os.listdir(folder_b):
    if file_name.replace('png','jpg') in files_in_a:
        src_path = os.path.join(folder_b, file_name.replace('jpg','png'))
        dst_path = os.path.join(folder_c, file_name)
        shutil.copy(src_path, dst_path)
        print(f"Copied: {file_name}")

print("All matching files have been copied.")