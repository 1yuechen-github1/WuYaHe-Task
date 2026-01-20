from networkx import center
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from utils import *  

# 主程序
if __name__ == "__main__":
    inp = r"C:\yuechen\code\wuyahe\1.code\2.data-缩放\screenshot\qianya-pca\img"
    outp = r"C:\yuechen\code\wuyahe\1.code\2.data-缩放\screenshot\pca-sum\img"
    os.makedirs(outp, exist_ok=True)
    os.makedirs(outp, exist_ok=True)
    total_files = 0
    success_files = 0
    for file in os.listdir(inp):
        ext = os.path.splitext(file)[1].lower()
        total_files += 1
        inp_file = os.path.join(inp, file)
        
        name, ext_orig = os.path.splitext(file)

        outp_file = os.path.join(outp, f"{name}{ext_orig}")
        # prex = file.split(".")[0]
        # prex = prex.split("_")[0]
        # os.makedirs(os.path.join(outp,prex), exist_ok=True)
        # outp_file = os.path.join(outp,prex, f"{name}{ext_orig}")
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
            if index is not None:
                sta,end = cacul_x(index,rot_img) 
                # (0, index), (wid-1, index) 起点终点  
                cv2.line(vis1, (sta, index), (end, index), (35, 147, 66), 1)
            ys, xs = np.where(rot_img > 0) 
            points = np.array(list(zip(xs, ys)))
            red_len = p1_rot[1] - index
            print(file,red_len,p1_rot,p2_rot,center_rot,index)
            # with open(f"{os.path.join(outp,prex)}"+'\len.txt', 'a') as f:
            with open(f"{outp}" + '\len.txt', 'a') as f:
                f.write(f"{file},{red_len}\n")
            cv2.line(vis1, tuple(p1_rot), tuple(p2_rot), (0, 0, 255), 3)
            cv2.circle(vis1, tuple(center_rot), 5, (255, 0, 0), -1)              
            # cv2.imshow("vis1",vis1)
            # cv2.waitKey(3000)            
            cv2.imencode('.jpg', vis1)[1].tofile(outp_file.replace(ext_orig,'.jpg'))

        


