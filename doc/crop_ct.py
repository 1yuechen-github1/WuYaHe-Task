# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# import argparse
# from pathlib import Path
# import numpy as np
# import nibabel as nib
# from nibabel.processing import resample_from_to


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--images-dir", required=True, help="imagesTr 路径（xxx_0000.nii.gz）")
#     parser.add_argument("--labels-dir", required=True, help="labelsTr 路径（xxx.nii.gz）")
#     parser.add_argument("--out-labels-dir", required=True, help="输出修正后的 labelsTr 路径")
#     args = parser.parse_args()

#     images_dir = Path(args.images_dir)
#     labels_dir = Path(args.labels_dir)
#     out_labels_dir = Path(args.out_labels_dir)
#     out_labels_dir.mkdir(parents=True, exist_ok=True)

#     img_files = sorted(images_dir.glob("*_0000.nii.gz"))
#     if not img_files:
#         img_files = sorted(images_dir.glob("*_0000.nii"))

#     ok, skip, miss, bad = 0, 0, 0, 0

#     for img_path in img_files:
#         name = img_path.name
#         case_id = name.replace("_0000.nii.gz", "").replace("_0000.nii", "")

#         # 匹配 label
#         lb_path = labels_dir / f"{case_id}.nii.gz"
#         if not lb_path.exists():
#             alt = labels_dir / f"{case_id}.nii"
#             if alt.exists():
#                 lb_path = alt
#             else:
#                 print(f"[MISS] {case_id}: label not found")
#                 miss += 1
#                 continue

#         try:
#             img_nii = nib.load(str(img_path))
#             lb_nii = nib.load(str(lb_path))

#             img_shape = img_nii.shape[:3]
#             lb_shape = lb_nii.shape[:3]

#             out_name = lb_path.name
#             out_path = out_labels_dir / out_name

#             if img_shape == lb_shape:
#                 # shape 已一致，直接复制（用 save 重写一份）
#                 lb_data = np.asanyarray(lb_nii.dataobj)
#                 nib.save(nib.Nifti1Image(lb_data, lb_nii.affine, lb_nii.header.copy()), str(out_path))
#                 print(f"[SKIP] {case_id}: shape already same {lb_shape}")
#                 skip += 1
#                 continue

#             # 按 image 网格重采样 label（最近邻）
#             lb_rs = resample_from_to(lb_nii, img_nii, order=0)  # order=0 for segmentation
#             lb_rs_data = np.asanyarray(lb_rs.dataobj).astype(np.int16, copy=False)

#             nib.save(nib.Nifti1Image(lb_rs_data, img_nii.affine, lb_nii.header.copy()), str(out_path))
#             print(f"[OK] {case_id}: label {lb_shape} -> {lb_rs_data.shape}, image={img_shape}")
#             ok += 1

#         except Exception as e:
#             print(f"[BAD] {case_id}: {e}")
#             bad += 1

#     print(f"\nDone. ok={ok}, skip={skip}, miss={miss}, bad={bad}")


# if __name__ == "__main__":
#     main()


import nibabel as nib
from pathlib import Path

img_dir = Path(r"\\Desktop-76khoer\d\1.CY-SPACE\WuYaHe\99-ct\data1-30\ct")
lab_dir = Path(r"\\Desktop-76khoer\d\1.CY-SPACE\WuYaHe\99-ct\data1-30\crop")

for p in sorted(img_dir.glob("*_0000.nii.gz")):
    
    cid = p.name.replace("_0000.nii.gz", "")
    l = lab_dir / f"{cid}.nii.gz"
    if not l.exists():
        print("missing label:", cid)
        continue
    si = nib.load(str(p)).shape
    sl = nib.load(str(l)).shape
    if si != sl:
        print("shape mismatch:", cid, si, sl)
