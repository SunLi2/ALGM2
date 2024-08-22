import numpy as np
#path to provided foreground images
fg_path = '/home/ljh/SLL/data/sll/MattingCode/ContextMatte/PATH/Adobe_Deep_Matting_Dataset/Combined_Dataset/Training_set/Other2/low_fg_t/'
# fg_path = "/home/ljh/SLL/data/sll/MattingCode/DataSet/Transparent-460/Val/val/"

# # path to provided alpha mattes
# # a_path = 'mask/'
a_path = "/home/ljh/SLL/data/sll/MattingCode/ContextMatte/PATH/Adobe_Deep_Matting_Dataset/Combined_Dataset/Training_set/Other2/low_alpha_t/"
# # Path to background images (MSCOCO)
bg_path = "/home/sll/data/sll/MattingCode/DataSet/train2014/"
#train_bg = "/home/Storage/sll/MattingCode/DataSet/train2014/"
# # Path to folder where you want the composited images to go
out_path = '/home/ljh/SLL/data/sll/MattingCode/ContextMatte/PATH/Adobe_Deep_Matting_Dataset/Combined_Dataset/Training_set/Other2/low_merged/'

##############################################################

from PIL import Image
import os
import torch
import cv2

def composite(fg, bg, alpha):
    alpha=alpha/255
    alpha = alpha.unsqueeze(0).permute(1,2,0)
    result=fg*alpha+bg*(1-alpha)
    # result = fg * alpha
    result=result.cpu().numpy()
    result=result.astype(np.uint8)
    return result

num_bgs = 20

fg_files = os.listdir(fg_path)
a_files = os.listdir(a_path)
bg_files = os.listdir(bg_path)

bg_iter = iter(bg_files)
for im_name in fg_files:

    im = cv2.imread(fg_path + im_name)
    if im_name in a_files:
        a = cv2.imread(a_path + im_name,0)
        im=torch.tensor(im)
        a=torch.tensor(a)
        im=im.cuda()
        a=a.cuda()

        bcount = 0
        for i in range(num_bgs):
            bg_name = next(bg_iter)
            bg = cv2.imread(bg_path + bg_name)
            bg_image = cv2.resize(bg, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_CUBIC)
            bg_image=torch.tensor(bg_image)
            bg_image=bg_image.cuda()
            out = composite(im, bg_image, a)
            cv2.imwrite(os.path.join(out_path, f'{im_name[:len(im_name) - 4]}_{bcount}.png'), out)
            # cv2.imwrite(os.path.join(out_path, im_name), out)
            print(im_name,bcount)
            bcount += 1