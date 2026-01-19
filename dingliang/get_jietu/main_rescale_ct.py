import os
import numpy as np
import nibabel as nib
from skimage.transform import rescale


def rescale_segmentation(input_dir, output_dir, target_spacing=1.0):
    """
    批量重采样分割标签图像到指定体素间距

    注意：分割标签必须使用 order=0（最近邻插值）
    以保持标签的离散性
    """
    os.makedirs(output_dir, exist_ok=True)

    for fname in os.listdir(input_dir):
        if not fname.endswith(('.nii', '.nii.gz')):
            continue

        # 加载数据
        img_path = os.path.join(input_dir, fname)
        img = nib.load(img_path)
        data = img.get_fdata()
        affine = img.affine.copy()
        header = img.header.copy()

        # 检查数据类型
        print(f"处理文件: {fname}")
        print(f"  原始数据类型: {data.dtype}")
        print(f"  原始值范围: [{np.min(data):.1f}, {np.max(data):.1f}]")
        print(f"  唯一标签值: {np.unique(data)}")

        # 获取原始间距
        orig_spacing = header.get_zooms()[:3]

        # 检查是否需要重采样
        if all(abs(s - target_spacing) < 1e-6 for s in orig_spacing):
            output_path = os.path.join(output_dir, fname)
            # 确保保存为整数类型
            if not np.issubdtype(data.dtype, np.integer):
                data = data.astype(np.int16)
            img = nib.Nifti1Image(data, affine, header)
            img.to_filename(output_path)
            print(f"  ✓ 间距已符合，转换为整数后保存")
            continue

        # 计算缩放因子
        scale = tuple(s / target_spacing for s in orig_spacing[::-1])  # 反转顺序

        # 重采样 - 分割标签必须使用 order=0
        resampled = rescale(data, scale, order=0,  # 最近邻插值
                            preserve_range=True,
                            mode='constant',  # 使用constant而不是reflect
                            cval=0,  # 边界填充0
                            anti_aliasing=False)

        # 转换为整数类型
        resampled = np.round(resampled).astype(np.int16)

        print(f"  重采样后数据类型: {resampled.dtype}")
        print(f"  重采样后值范围: [{np.min(resampled):.1f}, {np.max(resampled):.1f}]")
        print(f"  唯一标签值: {np.unique(resampled)}")

        # 更新仿射矩阵
        new_affine = affine.copy()
        for i in range(3):
            new_affine[:3, i] *= target_spacing / orig_spacing[i]

        # 创建新图像并保存
        new_img = nib.Nifti1Image(resampled, new_affine, header)
        output_path = os.path.join(output_dir, fname)
        nib.save(new_img, output_path)

        print(f"  ✓ 保存成功: {data.shape}→{resampled.shape}, 间距: {orig_spacing}→{target_spacing}")
        print("-" * 50)


# 使用示例
if __name__ == "__main__":
    rescale_segmentation(
        input_dir=r"C:\yuechen\code\wuyahe\1.code\0107\1030\dingliang\get_jietu\data\label",
        output_dir=r"C:\yuechen\code\wuyahe\1.code\0107\1030\dingliang\get_jietu\data\rescale",
        target_spacing=0.3
    )