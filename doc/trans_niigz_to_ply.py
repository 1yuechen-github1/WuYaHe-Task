# -*- coding: utf-8 -*-
import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.affines import apply_affine
from skimage.measure import marching_cubes


def save_ply_ascii(vertices: np.ndarray, faces: np.ndarray, out_path: Path) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for v in vertices:
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for tri in faces:
            f.write(f"3 {int(tri[0])} {int(tri[1])} {int(tri[2])}\n")


def nii_mask_to_ply(mask_path: Path, out_ply: Path, ct_path: Path | None = None,
                    label: int | None = None, iso: float = 0.5) -> None:
    mask_nii = nib.load(str(mask_path))
    mask_data = np.asarray(mask_nii.get_fdata())

    if label is None:
        vol = (mask_data > 0).astype(np.uint8)
    else:
        vol = (mask_data == label).astype(np.uint8)

    if vol.max() == 0:
        raise ValueError("mask 中没有可提取的体素（全 0）。")

    # 关键：用 CT 的 affine（若提供）保证和原始 CT 同一世界坐标系
    if ct_path is not None:
        ct_nii = nib.load(str(ct_path))
        ct_affine = ct_nii.affine
        if mask_nii.shape[:3] != ct_nii.shape[:3]:
            raise ValueError(f"mask/ct 尺寸不一致: {mask_nii.shape[:3]} vs {ct_nii.shape[:3]}")
        affine = ct_affine
    else:
        affine = mask_nii.affine

    # verts 是体素索引坐标 (i,j,k)
    verts_ijk, faces, _, _ = marching_cubes(vol, level=iso)
    # 转到世界坐标（mm），与 CT 对齐
    verts_xyz = apply_affine(affine, verts_ijk)
    # 按需求将 y 轴取反: y -> -y
    verts_xyz[:, 1] *= -1.0
    verts_xyz[:, 0] *= -1.0

    out_ply.parent.mkdir(parents=True, exist_ok=True)
    save_ply_ascii(verts_xyz, faces, out_ply)


def main():
    parser = argparse.ArgumentParser(description="将 mask(.nii/.nii.gz) 转为与 CT 对齐的 PLY")
    parser.add_argument("--mask", required=True, help="mask nii/nii.gz 路径")
    parser.add_argument("--out", required=True, help="输出 ply 路径")
    parser.add_argument("--ct", default=None, help="原始 CT nii/nii.gz 路径（建议传入）")
    parser.add_argument("--label", type=int, default=None, help="仅导出指定标签（默认 >0 全部）")
    parser.add_argument("--iso", type=float, default=0.5, help="marching cubes 阈值，默认 0.5")
    args = parser.parse_args()

    nii_mask_to_ply(
        mask_path=Path(args.mask),
        out_ply=Path(args.out),
        ct_path=Path(args.ct) if args.ct else None,
        label=args.label,
        iso=args.iso,
    )
    print(f"PLY 已保存: {args.out}")


if __name__ == "__main__":
    main()
