from networkx import center
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from utils import *  

# 主程序
if __name__ == "__main__":
    inp = r"C:\yuechen\code\wuyahe\1.code\2.data-缩放\screenshot\pca-sum\houya\img"
    outp = r"C:\yuechen\code\wuyahe\1.code\2.data-缩放\screenshot\pca-sum\houya\pca"
    os.makedirs(outp, exist_ok=True)
    os.makedirs(outp, exist_ok=True)
    total_files = 0
    success_files = 0
    for file in os.listdir(inp):
        ext = os.path.splitext(file)[1].lower()
        total_files += 1
        inp_file = os.path.join(inp, file)
        
        name, ext_orig = os.path.splitext(file)
        # outp_file = os.path.join(outp, f"{name}{ext_orig}")
        prex = file.split(".")[0]
        prex = prex.split("_")[0]
        os.makedirs(os.path.join(outp,prex), exist_ok=True)
        outp_file = os.path.join(outp,prex, f"{name}{ext_orig}")
        img = Image.open(inp_file)
        # pca 牙体长轴
        res= extract_tooth_long_axis(img, outp_file,file)
        if res:   
            rot_img,p1_rot,p2_rot,center_rot,vis = res
            vis1 = cv2.cvtColor(rot_img, cv2.COLOR_GRAY2BGR) 
            wid, heigh = rot_img.shape[1], rot_img.shape[0]
            # 5 + 7 = 12
            index = center_rot[1] - 12
            closest_point,min_y_point,_,_= cacul_y_by_x(p1_rot, p2_rot, rot_img,index)
            min_y_point1,max_y_point1 = get_axios1(rot_img,p1_rot,p2_rot)
            top_min, top_max = get_top_poin(rot_img, p1_rot, p2_rot)
            if index is not None:
                sta,end = cacul_x(index,rot_img)
                cv2.line(vis1, (sta, index), (end, index), (35, 147, 66), 1) # 2mm绿线
                cv2.line(vis1, closest_point, min_y_point, (35, 147, 66), 1) # 垂直长轴的绿线 下
                cv2.line(vis1, max_y_point1, min_y_point1, (35, 147, 66), 1) # 垂直长轴的绿线 上
                # cv2.circle(vis1, top_max, (35, 147, 66), 1, 3)
                # cv2.circle(vis1, tuple(top_max), 2, (255, 0, 0), -1)

            ys, xs = np.where(rot_img > 0) 
            points = np.array(list(zip(xs, ys)))
            red_len = np.sqrt((closest_point[0] - min_y_point[0]) ** 2 + (closest_point[1] - min_y_point[1]) ** 2)
            red_len1 = (max_y_point1[0] - top_max[0]) ** 2 + (max_y_point1[1] - top_max[1]) ** 2

            print(p1_rot[1],p2_rot[1],index)
            # with open(f"{outp}" + '\len.txt', 'a') as f:
            with open(f"{outp}\\{prex}" + '\len.txt', 'a') as f:
                f.write(f"{file},{red_len},{red_len1}\n")
            cv2.line(vis1, tuple(p1_rot), tuple(p2_rot), (0, 0, 255), 3)
            cv2.circle(vis1, tuple(center_rot), 5, (255, 0, 0), -1)
            cv2.putText(
                vis1,
                text=f'a = {red_len * 0.3:.2f}mm',
                org=(50, 50),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(0, 0, 255),
                thickness=2,
                lineType=cv2.LINE_AA
            )

            # cv2.imshow("vis1",vis1)
            # cv2.waitKey(3000)            
            cv2.imencode('.jpg', vis1)[1].tofile(outp_file.replace(ext_orig,'.jpg'))

        


