import os
import nibabel as nib
import numpy as np

def remap(path):
    for file in os.listdir(path):
        if not file.endswith(".nii") and not file.endswith(".nii.gz"):
            continue

        file_path = os.path.join(path, file)
        out_path = os.path.join(r"F:\wuyahe\data\1-30\remap", file)
        print("处理:", file)

        nii = nib.load(file_path)
        data = nii.get_fdata()  # 读取为 numpy array

        # ⚠️ 建议转 int（label 一般是整数）
        data = data.astype(np.int16)

        # ===== 标签映射 =====
        data[data == 3] = 2
        data[data == 4] = 3

        # ===== 重新保存 =====
        new_nii = nib.Nifti1Image(data, nii.affine, nii.header)
        nib.save(new_nii, file_path)  # 覆盖原文件（如果不想覆盖可以改路径）

        print("完成:", file, "\n")

remap(r"\\Desktop-76khoer\d\1.CY-SPACE\WuYaHe\99-ct\data1-30\label")