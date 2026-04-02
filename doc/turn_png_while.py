import cv2
import numpy as np
import os

def fill_hole(path):
    for file in os.listdir(path):
        if file.endswith('.png'):
            print(file)
            # 读取图像
            img = cv2.imread(os.path.join(path, file))
            # 复制一份用于 floodFill 操作
            im_floodfill = img.copy()
            # 获取图像尺寸
            h, w = img.shape[:2]
            # floodFill 需要的 mask（尺寸必须比原图大2）
            mask = np.zeros((h + 2, w + 2), np.uint8)
            print('img:', img.shape)
            # 找到图像中蓝色区域（BGR格式：255,0,0）
            # 返回 bool mask
            blue_mask = np.all(img == [255, 0, 0], axis=-1)
            print('blue_mask:', blue_mask.sum())
            # 从左上角开始 floodFill
            # 作用：把背景区域全部填成 255
            cv2.floodFill(im_floodfill, mask, (0, 0), 255)
            # 对 floodFill 结果取反
            # 得到原图内部的空洞区域
            im_floodfill_inv = cv2.bitwise_not(im_floodfill)
            # 原图 和 空洞区域 合并
            # 实现填充空洞
            out = img | im_floodfill_inv

            # 再复制一份结果
            result = out.copy()

            # 把原来的蓝色区域恢复
            # 防止 floodFill 影响原有标注
            result[blue_mask] = img[blue_mask]

            yellow_mask = np.all(result == [0, 255, 255], axis=-1)
            print('yellow_mask:', yellow_mask.sum())
            result[yellow_mask] = [0, 0, 0]
            
            # 保存结果
            cv2.imwrite(os.path.join(r'data\mid\processed\test\3', file), result)

fill_hole(r'data\mid\origin\test\3')