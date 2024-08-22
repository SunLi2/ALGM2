import os
import cv2
import numpy as np
import pandas as pd
from evaluate import compute_sad_loss, compute_mse_loss, compute_connectivity_error, compute_gradient_loss
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred-dir', type=str,
                        default='/home/Storage/sll/MattingCode/matteformer_fg_alpha/predDIM/231129_221117/',
                        help="output dir")
    parser.add_argument('--label-dir', type=str,
                        default='/home/Storage/sll/MattingCode/DataSet/Transparent-460/Test/alpha_copy/',
                        help="GT alpha dir")
    parser.add_argument('--trimap-dir', type=str,
                        default='/home/Storage/sll/MattingCode/DataSet/Transparent-460/Test/trimap_copy/',
                        help="trimap dir")
    args = parser.parse_args()

    mse_loss = []
    sad_loss = []
    mse_loss_unknown = []
    sad_loss_unknown = []
    grad_loss = []
    compute_connectivity_loss = []
    grad_loss_unknown = []
    compute_connectivity_loss_unknown = []

    image_names = []  # 新增一个列表用于存储图像名字

    for img in os.listdir(args.pred_dir):
        image_names.append(img)  # 记录当前图像名字

        label = cv2.imread(os.path.join(args.label_dir, img), 0).astype(np.float32)
        pred = cv2.imread(os.path.join(args.pred_dir, img), 0).astype(np.float32)
        trimap = cv2.imread(os.path.join(args.trimap_dir, img), 0).astype(np.float32)
        if pred.shape != label.shape:
            pred = cv2.resize(pred, (label.shape[1], label.shape[0]))

        mse_loss_unknown_ = compute_mse_loss(pred, label, trimap)
        sad_loss_unknown_ = compute_sad_loss(pred, label, trimap)[0]
        gradient_loss_unknown = compute_gradient_loss(pred, label, trimap)
        connectivity_loss_unknown = compute_connectivity_error(pred, label, trimap, 0.1)

        trimap[...] = 128
        mse_loss_ = compute_mse_loss(pred, label, trimap)
        sad_loss_ = compute_sad_loss(pred, label, trimap)[0]
        gradient_loss = compute_gradient_loss(pred, label, trimap)
        connectivity_loss = compute_connectivity_error(pred, label, trimap, 0.1)

        print('Whole Image: MSE: ', mse_loss_, ' SAD:', sad_loss_, "GRAD:", gradient_loss, "Conn:", connectivity_loss)
        print('Unknown Region: MSE:', mse_loss_unknown_, ' SAD:', sad_loss_unknown_, "GRAD:", gradient_loss_unknown,
              "Conn:", connectivity_loss_unknown)

        mse_loss_unknown.append(mse_loss_unknown_)
        sad_loss_unknown.append(sad_loss_unknown_)
        mse_loss.append(mse_loss_)
        sad_loss.append(sad_loss_)
        grad_loss.append(gradient_loss)
        compute_connectivity_loss.append(connectivity_loss)
        grad_loss_unknown.append(gradient_loss_unknown)
        compute_connectivity_loss_unknown.append(connectivity_loss_unknown)

    # 创建包含损失信息和图像名字的 DataFrame
    data = {
        'Image_Name': image_names,
        'MSE_whole': mse_loss,
        'SAD_whole': sad_loss,
        'GRAD_whole': grad_loss,
        'CONN_whole': compute_connectivity_loss,
        'MSE_unknown': mse_loss_unknown,
        'SAD_unknown': sad_loss_unknown,
        'GRAD_unknown': grad_loss_unknown,
        'CONN_unknown': compute_connectivity_loss_unknown
    }

    df = pd.DataFrame(data)

    # 保存 DataFrame 到 Excel 文件
    output_excel_path = '/home/Storage/sll/MattingCode/matteformer_fg_alpha/excel/231129_221117.xlsx'
    df.to_excel(output_excel_path, index=False)

    print(f"Results saved to {output_excel_path}")
