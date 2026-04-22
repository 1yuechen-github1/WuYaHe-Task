from networkx import center
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from utils import *  
from tqdm import tqdm, trange


PIXEL_SPACING_MM = 0.3


def draw_horizontal_dashed_line(img, y, x0, x1, color, thickness=1, dash=8, gap=6):
    y = int(y)
    x0, x1 = int(min(x0, x1)), int(max(x0, x1))
    x = x0
    while x <= x1:
        x_end = min(x + dash, x1)
        cv2.line(img, (x, y), (x_end, y), color, thickness, lineType=cv2.LINE_AA)
        x += dash + gap

# 主程序
if __name__ == "__main__":
    inp = r"C:\yuechen\code\wuyahe\1.code\0212\output\base\screenshot\qianya"
    outp = r"C:\yuechen\code\wuyahe\1.code\0212\output\pca\pca-qianya"
    os.makedirs(outp, exist_ok=True)
    os.makedirs(outp, exist_ok=True)
    total_files = 0
    success_files = 0
    # for file in os.listdir(inp):
    for file in tqdm(os.listdir(inp), desc='对后牙截图定量', unit='file'):    
        for file1 in os.listdir(os.path.join(inp,file)):
            ext = os.path.splitext(file)[1].lower()
            total_files += 1
            inp_file = os.path.join(inp, file,file1)

            name, ext_orig = os.path.splitext(file1)

            # outp_file = os.path.join(outp, f"{name}{ext_orig}")
            prex = file.split(".")[0]
            prex = prex.split("_")[0]
            os.makedirs(os.path.join(outp,prex), exist_ok=True)
            outp_file = os.path.join(outp,prex, f"{name}{ext_orig}")
            img = pil_imread(inp_file)  
            # pca 牙体长轴
            res= extract_tooth_long_axis(img, outp_file,file)
            if res:   
                rot_img,p1_rot,p2_rot,center_rot,vis = res
                vis1 = cv2.cvtColor(rot_img, cv2.COLOR_GRAY2BGR) 
                wid, heigh = rot_img.shape[1], rot_img.shape[0]
                # 本院是000开头， spacing = 0.3 6 / 0.3 = 20
                # 500 开头 spacing = 0.25/0.35

                id = file.split("_")[0]
                index = get_axiosx(rot_img, dist = 20)
                index_top = get_axiosx_top(rot_img, dist=20)
                if index is not None and index_top is not None:
                    sta,end = cacul_x(index,rot_img)
                    sta_top, end_top = cacul_x(index_top, rot_img)
                    cv2.line(vis1, (sta, index), (end, index), (35, 147, 66), 1, lineType=cv2.LINE_AA)
                    cv2.line(vis1, (sta_top, index_top), (end_top, index_top), (35, 147, 66), 1, lineType=cv2.LINE_AA)
                    ys, xs = np.where(rot_img > 0)
                    if len(ys) == 0:
                        continue
                    y_top = int(np.min(ys))
                    sta_dash, end_dash = cacul_x(y_top, rot_img)
                    draw_horizontal_dashed_line(
                        vis1, y_top, sta_dash - 5, end_dash + 5, (35, 147, 66), thickness=1
                    )
                    a = abs(index_top - index)
                    b = abs(y_top - index_top)
                # else:
                #     continue

                # print(file, a, b, p1_rot, p2_rot, center_rot, index)
                # with open(f"{outp}" + '\len.txt', 'a') as f:
                with open(os.path.join(outp, prex, "len.txt"), 'a') as f:
                    f.write(f"{file},{a * 0.3},{b * 0.3},{a+b}\n")
                cv2.line(vis1, tuple(p1_rot), tuple(p2_rot), (0, 0, 255), 3, lineType=cv2.LINE_AA)
                cv2.circle(vis1, tuple(center_rot), 5, (255, 0, 0), -1)
                cv2.putText(
                    vis1,
                    text=f'a = {a * PIXEL_SPACING_MM:.2f}mm',
                    org=(50, 50),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color=(0, 0, 255),
                    thickness=2,
                    lineType=cv2.LINE_AA
                )
                cv2.putText(
                    vis1,
                    text=f'b = {b * PIXEL_SPACING_MM:.2f}mm',
                    org=(50, 95),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color=(35, 147, 66),
                    thickness=2,
                    lineType=cv2.LINE_AA
                )

                # cv2.imshow("vis1",vis1)
                # cv2.waitKey(3000)      
                # print('outp_file:',outp_file.replace(ext_orig,'.jpg'))      
                cv2.imencode('.png', vis1)[1].tofile(outp_file.replace(ext_orig,'.png'))

        

