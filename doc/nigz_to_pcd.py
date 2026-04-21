# -*- coding: utf-8 -*-
import os
from pathlib import Path

import nibabel as nib
import numpy as np


def nifti_to_pointcloud(fdata, affine, label_value=1):
    """
    把 mask 中指定标签转为世界坐标点云 (N,3)
    """
    label_coords = np.argwhere(fdata == label_value)  # voxel ijk
    if label_coords.size == 0:
        return np.empty((0, 3), dtype=np.float64)

    # 可选：用体素中心，减少半个体素偏差
    label_coords = label_coords.astype(np.float64) + 0.5

    homogeneous_coords = np.c_[label_coords, np.ones(len(label_coords))]
    world_coords = (affine @ homogeneous_coords.T).T[:, :3]  # mm
    # 按需求将 y 轴取反: y -> -y
    world_coords[:, 1] *= -1.0
    return world_coords


def save_xyz_txt(points_xyz, out_txt):
    np.savetxt(out_txt, points_xyz, fmt="%.6f %.6f %.6f")


def convert_folder(mask_dir, out_dir, label_value=1):
    mask_dir = Path(mask_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    nii_files = [p for p in mask_dir.iterdir() if p.name.endswith(".nii") or p.name.endswith(".nii.gz")]
    if not nii_files:
        print("没找到 nii/nii.gz 文件")
        return

    for p in nii_files:
        img = nib.load(str(p))
        fdata = np.asarray(img.get_fdata())
        affine = img.affine

        points = nifti_to_pointcloud(fdata, affine, label_value=label_value)
        out_txt = out_dir / f"{p.name.replace('.nii.gz', '').replace('.nii', '')}_label{label_value}.txt"

        if len(points) == 0:
            print(f"[跳过] {p.name}: 没有 label={label_value}")
            continue

        save_xyz_txt(points, out_txt)
        print(f"[完成] {p.name} -> {out_txt.name}, 点数={len(points)}")


if __name__ == "__main__":
    mask_path = r"C:\yuechen\code\wuyahe\1.code\0212\data\monizhongzhi\labelsTr"
    output_path = r"C:\yuechen\code\wuyahe\1.code\0212\data\monizhongzhi\pcd_labels"
    convert_folder(mask_path, output_path, label_value=1)
