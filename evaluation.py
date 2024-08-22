import os
import cv2
import numpy as np
import pandas as pd  # 引入 pandas 库
from utils import compute_sad_loss, compute_mse_loss
import argparse


def evaluate(args):
    img_names = []
    mse_loss_unknown = []
    sad_loss_unknown = []

    for i, img in enumerate(os.listdir(args.label_dir)):
        if not((os.path.isfile(os.path.join(args.pred_dir, img)) and
                os.path.isfile(os.path.join(args.label_dir, img)) and
                os.path.isfile(os.path.join(args.trimap_dir, img)))):
            print('[{}/{}] "{}" skipping'.format(i, len(os.listdir(args.label_dir)), img))
            continue

        pred = cv2.imread(os.path.join(args.pred_dir, img), 0).astype(np.float32)
        label = cv2.imread(os.path.join(args.label_dir, img), 0).astype(np.float32)
        trimap = cv2.imread(os.path.join(args.trimap_dir, img), 0).astype(np.float32)

        # calculate loss
        mse_loss_unknown_ = compute_mse_loss(pred, label, trimap)
        sad_loss_unknown_ = compute_sad_loss(pred, label, trimap)[0]
        print('Unknown Region: MSE:', mse_loss_unknown_, ' SAD:', sad_loss_unknown_)

        # save for average
        img_names.append(img)
        mse_loss_unknown.append(mse_loss_unknown_)  # mean l2 loss per unknown pixel
        sad_loss_unknown.append(sad_loss_unknown_)  # l1 loss on unknown area

        print('[{}/{}] "{}" processed'.format(i, len(os.listdir(args.label_dir)), img))

    # 将结果保存到 Excel 文件
    results_df = pd.DataFrame({
        'Image Name': img_names,
        'MSE Loss Unknown': mse_loss_unknown,
        'SAD Loss Unknown': sad_loss_unknown
    })

    results_df.to_excel('evaluation_results.xlsx', index=False)

    print('* Unknown Region: MSE:', np.array(mse_loss_unknown).mean(), ' SAD:', np.array(sad_loss_unknown).mean())
    print('* Evaluation results saved to "evaluation_results.xlsx"')
    print('* If you want to report scores in your paper, please use the official MATLAB codes for evaluation.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred-dir', type=str,
                        default='/home/ljh/SLL/data/sll/MattingCode/AAAA/matteformer_fg_alpha/preDIM/trimap_composition/240519_104600_alpha/',
                        help="output dir")
    parser.add_argument('--label-dir', type=str,
                        default='/home/ljh/SLL/data/sll/MattingCode/ContextMatte/PATH/Adobe_Deep_Matting_Dataset/Combined_Dataset/Composition-1k-testset/alpha_copy/',
                        help="GT alpha dir")
    parser.add_argument('--trimap-dir', type=str,
                        default='/home/ljh/SLL/data/sll/MattingCode/ContextMatte/PATH/Adobe_Deep_Matting_Dataset/Combined_Dataset/Composition-1k-testset/trimaps/',
                        help="trimap dir")
    args = parser.parse_args()

    evaluate(args)
