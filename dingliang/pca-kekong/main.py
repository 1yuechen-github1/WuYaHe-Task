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
    inp = r"C:\yuechen\code\wuyahe\1.code\2.data-缩放\screenshot1\kekong"
    outp = r"C:\yuechen\code\wuyahe\1.code\2.data-缩放\screenshot1\pca\kekong-pca"

    os.makedirs(outp, exist_ok=True)

    total_files = 0
    success_files = 0
    # kekong_dir_left = os.path.join(outp, "left")
    # kekong_dir_right = os.path.join(outp, "right")
    for file in tqdm(os.listdir(inp), desc="processing", unit="file"):
    # for file in os.listdir(inp):
        single_file = os.path.join(inp, file)
        files = sorted(
            [f for f in os.listdir(single_file) if f.endswith(".png")],
            key=lambda x: int(re.search(r"slice_(\d+)", x).group(1)),
        )
        for file2 in files:
            # print(file2)
            ext = os.path.splitext(file2)[1].lower()
            total_files += 1
            inp_file = os.path.join(inp, file, file2)

            name, ext_orig = os.path.splitext(file2)
            prex = file2.split(".")[0]
            prex = prex.split("_")[0]
            os.makedirs(os.path.join(outp, prex), exist_ok=True)
            outp_file = os.path.join(outp, prex, f"{name}{ext_orig}")

            img = Image.open(inp_file)
            rot_img, p1_rot, p2_rot, center_rot, vis, blue_points = extract_tooth_long_axis(
                img, outp_file, file2, offset=10
            )

            vis1 = cv2.cvtColor(rot_img, cv2.COLOR_GRAY2BGR)

            offset_2mm_px = mm_to_px(2.0)
            target_width_px = mm_to_px(6.0)

            if len(blue_points) == 0:
                with open(f"{outp}\\{prex}" + "\\len.txt", "a") as f:
                    f.write(f"{file2},{0:.2f}\n")
                print(f"{file2}: no blue points found")
                continue

            blue_top_index = int(np.round(np.min(blue_points[:, 1])))
            blue_top_index = int(np.clip(blue_top_index, 0, rot_img.shape[0]))

            blue_dashed_line = get_blue_dashed_line(rot_img, p1_rot, p2_rot, blue_points)
            if blue_dashed_line is None:
                with open(f"{outp}\\{prex}" + "\\len.txt", "a") as f:
                    f.write(f"{file2},{0:.2f}\n")
                print(f"{file2}: failed to build blue dashed line")
                continue

            blue_solid_line = get_perp_line_by_axis_offset(
                rot_img,
                p1_rot,
                p2_rot,
                blue_dashed_line["closest_point"],
                offset_2mm_px,
            )
            if blue_solid_line is None:
                with open(f"{outp}\\{prex}" + "\\len.txt", "a") as f:
                    f.write(f"{file2},{0:.2f}\n")
                print(f"{file2}: failed to build blue solid line")
                continue
            # print('target_width_px:',target_width_px)
            green_solid_line = find_first_width_perp_line(
                rot_img, p1_rot, p2_rot, blue_solid_line["closest_point"][1], target_width_px
            )
            if green_solid_line is None:
                print(f"{file2}: failed to find target 6mm line")
                continue

            draw_line_by_style(vis1, blue_dashed_line, (255, 0, 0), dashed=True, thickness=1)
            draw_line_by_style(vis1, blue_solid_line, (255, 0, 0), dashed=False, thickness=1)
            draw_line_by_style(vis1, green_solid_line, (0, 255, 0), dashed=False, thickness=1)

            a_len = (
                np.linalg.norm(
                    np.array(blue_solid_line["closest_point"]) - np.array(green_solid_line["closest_point"])
                )
                * PIXEL_SPACING_MM
            )
            b_len = 0.0

            with open(f"{outp}\\{prex}" + "\\len.txt", "a") as f:
                f.write(f"{file2},{a_len:.2f}\n")

            cv2.line(vis1, tuple(p1_rot), tuple(p2_rot), (0, 0, 255), 3, lineType=cv2.LINE_AA)
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

            cv2.imencode(".png", vis1)[1].tofile(outp_file.replace(ext_orig, ".png"))
            success_files += 1
