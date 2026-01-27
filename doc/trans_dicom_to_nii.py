
import os
import numpy as np
import SimpleITK as sitk


# ---------------------------
# CT 窗口化
# ---------------------------
# def window_ct(image, window, width):
#     lower = window - width / 2
#     upper = window + width / 2
#     image = sitk.Clamp(image, lower, upper)
#     image = sitk.RescaleIntensity(image, 0, 255)
#     image = sitk.Cast(image, sitk.sitkUInt8)
#     return image
def window_ct(image, window, width):
    lower = window - width / 2
    upper = window + width / 2

    image = sitk.Clamp(
        image,
        sitk.sitkInt16,  # 👈 必须指定输出像素类型
        lower,
        upper
    )

    image = sitk.RescaleIntensity(image, 0, 255)
    image = sitk.Cast(image, sitk.sitkUInt8)
    return image


# ---------------------------
# MIP + 窗口
# ---------------------------
def mip_with_window(image, axis=0, window=1500, width=1700):
    arr = sitk.GetArrayFromImage(image)  # (z, y, x)

    mip = np.max(arr, axis=axis)

    lower = window - width / 2
    upper = window + width / 2
    mip = np.clip(mip, lower, upper)

    mip = (mip - lower) / (upper - lower)
    mip = (mip * 255).astype(np.uint8)

    mip_img = sitk.GetImageFromArray(mip)
    mip_img.SetSpacing(image.GetSpacing()[1:])
    return mip_img


# ---------------------------
# DICOM → NIfTI（全流程）
# ---------------------------
def dicom_to_nii_full(
    dicom_root,
    output_root,
    ct_window=1000,
    ct_width=4000,
    mip_window=1500,
    mip_width=1700
):
    os.makedirs(output_root, exist_ok=True)

    for folder in os.listdir(dicom_root):
        dicom_dir = os.path.join(dicom_root, folder)
        if not os.path.isdir(dicom_dir):
            continue

        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)
        if not dicom_names:
            print(f"[跳过] {dicom_dir} 没有 DICOM")
            continue

        reader.SetFileNames(dicom_names)
        img = reader.Execute()

        # 1️⃣ 原始 HU
        # raw_path = os.path.join(output_root, f"{folder}.nii.gz")
        # sitk.WriteImage(img, raw_path)

        # 2️⃣ CT 窗口化
        ct_vis = window_ct(img, ct_window, ct_width)
        ct_vis_path = os.path.join(
            output_root, f"Wuyahe_{folder}_0000.nii.gz"
        )
        sitk.WriteImage(ct_vis, ct_vis_path)

        # 3️⃣ MIP
        # mip_img = mip_with_window(
        #     img,
        #     axis=0,  # Z 轴 MIP（可改）
        #     window=mip_window,
        #     width=mip_width
        # )
        # mip_path = os.path.join(
        #     output_root, f"{folder}_mip_w{mip_window}_w{mip_width}.nii.gz"
        # )
        # sitk.WriteImage(mip_img, mip_path)

        print(f"[完成] {folder}")


# ---------------------------
# 使用示例
# ---------------------------
dicom_to_nii_full(
    dicom_root=r"C:\yuechen\code\wuyahe\2.data\0122\wash\dicom",
    output_root=r"C:\yuechen\code\wuyahe\2.data\0122\wash\nii1",
    ct_window=1000,
    ct_width=4000,
    mip_window=1500,
    mip_width=1700
)

