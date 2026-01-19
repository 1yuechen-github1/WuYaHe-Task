# 1. 提取spacing
import os
import nibabel as nib

def get_spac(dir):
    files = os.listdir(dir)
    for file in files:
        path = os.path.join(dir,file)
        spac = nib.load(path).header.get_zooms()
        print(file,spac)

get_spac(r"F:\wuyahe\第二第三批截图\xiaqianyaCT\wuayhe 第二批\3.nii\data\label")

# 2. 提取affine