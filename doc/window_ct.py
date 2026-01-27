import os

import nibabel as nib
import numpy as np

# 加载CT和分割
ct_path = r'C:\yuechen\code\wuyahe\1.code\2.data-缩放\wash\ct'
seg_path = r'C:\yuechen\code\wuyahe\1.code\2.data-缩放\wash\label'
for file in os.listdir(ct_path):
    ct_img = nib.load(os.path.join(ct_path, file))
    seg_img = nib.load(os.path.join(seg_path, file))
    ct_data = ct_img.get_fdata()
    seg_data = seg_img.get_fdata()
    seg_mask = seg_data > 0
    # 在原CT值基础上增加固定值
    HU_increase = 3000  # 增加1000 HU
    enhanced_ct = ct_data.copy()
    enhanced_ct[seg_mask] = ct_data[seg_mask] + HU_increase
    # 保存
    new_img = nib.Nifti1Image(enhanced_ct, ct_img.affine, ct_img.header)
    nib.save(new_img, f'C:\\yuechen\\code\\wuyahe\\1.code\\2.data-缩放\\wash\\wash\\{file}')

