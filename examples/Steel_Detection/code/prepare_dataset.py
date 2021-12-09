import os
import numpy as np
import pandas as pd
import shutil
import cv2


def name_and_mask(start_idx):
    '''
    获取文件名和mask
    '''
    col = start_idx
    img_names = [
        str(i).split("_")[0] for i in train_df.iloc[col:col + 4, 0].values
    ]
    if not (img_names[0] == img_names[1] == img_names[2] == img_names[3]):
        raise ValueError

    labels = train_df.iloc[col:col + 4, 1]
    mask = np.zeros((256, 1600), dtype=np.uint8)
    mask_label = np.zeros(1600 * 256, dtype=np.uint8)

    for idx, label in enumerate(labels.values):
        if label is not np.nan:
            label = label.split(" ")
            positions = map(int, label[0::2])
            length = map(int, label[1::2])
            for pos, le in zip(positions, length):
                mask_label[pos - 1:pos + le - 1] = idx + 1

    mask[:, :] = mask_label.reshape(256, 1600, order='F')  #按列取值reshape
    return img_names[0], mask


if __name__ == '__main__':
    # 创建数据集目录结构
    target_root = "steel"
    if os.path.exists(target_root):
        shutil.rmtree(target_root)
    os.makedirs(target_root)

    target_ann = os.path.join(target_root, "Annotations")
    if os.path.exists(target_ann):
        shutil.rmtree(target_ann)
    os.makedirs(target_ann)

    target_image = os.path.join(target_root, "JPEGImages")
    if os.path.exists(target_image):
        shutil.rmtree(target_image)
    os.makedirs(target_image)

    # 原始数据集图像目录
    train_path = "severstal/train_images"

    # 读取csv文本数据
    train_df = pd.read_csv("severstal/train.csv")

    # 逐个图像生成mask
    index = 1
    for col in range(0, len(train_df), 4):
        img_names = [
            str(i).split("_")[0] for i in train_df.iloc[col:col + 4, 0].values
        ]
        if not (img_names[0] == img_names[1] == img_names[2] == img_names[3]):
            raise ValueError

        name, mask = name_and_mask(col)

        # 拷贝img(jpg格式)
        src_path = os.path.join(train_path, name)
        dst_path = os.path.join(target_image, name)
        shutil.copyfile(src_path, dst_path)

        # 写入标注文件(png格式)
        dst_path = os.path.join(target_ann, name.split('.')[0] + '.png')
        cv2.imwrite(dst_path, mask)

        print('完成第 ' + str(index) + '张图片')
        index += 1

    print('全部完成')
