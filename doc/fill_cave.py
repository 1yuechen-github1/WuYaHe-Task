import os
import cv2
import numpy as np


def cv_show(img):
    """显示图像"""
    cv2.imshow('', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def fill_cave(path):
    """
    批量处理文件夹中的图片：
    1. 将非白色区域膨胀、腐蚀填洞
    2. 保留白色背景
    """
    kernel = np.ones((3, 3), dtype=np.uint8)

    # 输出路径
    out_path = r'C:\Users\yuechen\Desktop\linshi\while\0'
    os.makedirs(out_path, exist_ok=True)

    for file in os.listdir(path):
        img_path = os.path.join(path, file)

        # 读取图片
        img = cv2.imdecode(
            np.fromfile(img_path, dtype=np.uint8),
            cv2.IMREAD_COLOR
        )

        # mask: True 表示非白色区域
        mask = np.any(img != 255, axis=2)

        # 分离非白色和白色区域
        img1 = np.zeros_like(img)
        img1[mask] = img[mask]

        img2 = np.zeros_like(img)
        img2[~mask] = img[~mask]

        # 膨胀 + 腐蚀处理非白色区域
        dilate = cv2.dilate(img1, kernel, iterations=4)
        dilate = cv2.erode(dilate, kernel, iterations=2)

        # 合并白色区域
        result = cv2.add(dilate, img2)

        # 保存图片
        cv2.imwrite(os.path.join(out_path, file), result)


if __name__ == '__main__':
    fill_cave(r'C:\Users\yuechen\Desktop\linshi\0')