# import tqdm
from tqdm import tqdm, trange

import cv2
import numpy as np
import os
from PIL import Image
from utils import *
import re


PIXEL_SPACING_MM = 0.3
dot_radius = 3


if __name__ == "__main__":
    # 1. 蓝色虚线位于颏孔上界，垂直于长轴
    # 2. 蓝色虚线向上平移 2mm 得到蓝色实线，垂直于长轴
    # 3. 蓝色实线继续向上平移，找到 mask>0 区域长度约为 7mm 的绿色实线，垂直于长轴
    # 4. 绿色实线继续向上平移，找到与 mask 上边界相切的绿色虚线，垂直于长轴
    # 5. 计算蓝色实线与绿色实线之间的距离 a
    # 6. 计算绿色实线与绿色虚线之间的距离 b

    inp = r"C:\yuechen\code\wuyahe\1.code\2.data-缩放\screenshot\houya"
    outp = r"C:\yuechen\code\wuyahe\1.code\2.data-缩放\screenshot\pca\pca-houya"

    os.makedirs(outp, exist_ok=True)

    total_files = 0
    success_files = 0

    # for file in os.listdir(inp):
    for file in tqdm(os.listdir(inp), desc='对后牙截图定量', unit='file'):
        single_file = os.path.join(inp, file)
        files = sorted(
                [f for f in os.listdir(single_file) if f.endswith(".png")],
                key=lambda x: int(re.search(r"slice_(\d+)", x).group(1)),
            )
        for file2 in files:
            
            ext = os.path.splitext(file2)[1].lower()
            total_files += 1
            inp_file = os.path.join(inp,file, file2)

            name, ext_orig = os.path.splitext(file2)
            prex = file2.split(".")[0]
            prex = prex.split("_")[0]
            os.makedirs(os.path.join(outp, prex), exist_ok=True)
            outp_file = os.path.join(outp, prex, f"{name}{ext_orig}")

            img = Image.open(inp_file)
            rot_img, p1_rot, p2_rot, center_rot, vis, blue_points = extract_tooth_long_axis(img, outp_file, file2, offset=10)
            # if not res:
            #     continue

            # rot_img, p1_rot, p2_rot, center_rot, vis, blue_points = res
            vis1 = cv2.cvtColor(rot_img, cv2.COLOR_GRAY2BGR)

            offset_2mm_px = mm_to_px(2.0)
            target_width_px = mm_to_px(7.0)

            # 蓝色区域上界作为第一条蓝色虚线的参考位置
            if len(blue_points) > 0:
                blue_top_index = int(np.round(np.min(blue_points[:, 1])))
            # else:
            #     blue_top_index = int(center_rot[1])


            blue_top_index = int(np.clip(blue_top_index, 0, rot_img.shape[0] ))
            blue_dashed_line = get_index_perp_line(rot_img, p1_rot, p2_rot, blue_top_index,blue_points)
            if blue_dashed_line is None:
                print(f"{file2}: 未找到蓝色虚线")
                continue

            # 蓝色虚线向上平移 2mm，得到蓝色实线
            blue_solid_index = max(0, blue_top_index - offset_2mm_px)
            blue_solid_line = get_perp_line_at_index(rot_img, p1_rot, p2_rot, blue_solid_index)
            if blue_solid_line is None:
                print(f"{file2}: 未找到蓝色实线")
                continue

            # 继续向上搜索，找到长度最接近 7mm 的绿色实线
            green_solid_line = find_best_width_perp_line(
                rot_img, p1_rot, p2_rot, blue_solid_index, target_width_px
            )
            if green_solid_line is None:
                print(f"{file2}: 未找到满足条件的 7mm 绿色实线")
                continue

            # 在绿色实线上方继续找与 rot_img>0 只有一个像素交点的绿色虚线
            green_dashed_line = find_upper_tangent_perp_line(
                rot_img, p1_rot, p2_rot, green_solid_line["index"] -3
            )
            if green_dashed_line is None:
                print(f"{file2}: 未找到仅有一个像素交点的绿色虚线")
                continue

            # 绘制四条关键参考线
            draw_line_by_style(vis1, blue_dashed_line, (255, 0, 0), dashed=True, thickness=1)
            draw_line_by_style(vis1, blue_solid_line, (255, 0, 0), dashed=False, thickness=1)
            draw_line_by_style(vis1, green_solid_line, (0, 255, 0), dashed=False, thickness=1)
            draw_line_by_style(vis1, green_dashed_line, (0, 255, 0), dashed=True, thickness=1)

            # a 是蓝色实线到绿色实线在长轴方向上的距离
            a_len = (
                np.linalg.norm(
                    np.array(blue_solid_line["closest_point"]) - np.array(green_solid_line["closest_point"])
                )
                * PIXEL_SPACING_MM
            )
            # b 是绿色实线到绿色虚线在长轴方向上的距离
            b_len = (
                np.linalg.norm(
                    np.array(green_solid_line["closest_point"]) - np.array(green_dashed_line["closest_point"])
                )
                * PIXEL_SPACING_MM
            )

            with open(f"{outp}\\{prex}" + "\\len.txt", "a") as f:
                f.write(
                    f"{file2},{a_len:.2f},{b_len:.2f}\n"
                )

            cv2.line(vis1, tuple(p1_rot), tuple(p2_rot), (0, 0, 255), 3)
            for pt in blue_points:
                x, y = int(pt[0]), int(pt[1])
                vis1[y, x] = (255, 0, 0)


            cv2.putText(
                vis1,
                text=f"a = {a_len:.2f}mm",
                org=(50, 50),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(255, 0, 0),
                thickness=2,
                lineType=cv2.LINE_AA,
            )
            cv2.putText(
                vis1,
                text=f"b = {b_len:.2f}mm",
                org=(50, 95),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(0, 255, 0),
                thickness=2,
                lineType=cv2.LINE_AA,
            )

            cv2.imencode(".jpg", vis1)[1].tofile(outp_file.replace(ext_orig, ".jpg"))
            success_files += 1
