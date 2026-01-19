import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
  
# 设置中文字体和输出路径
# plt.rcParams['font.sans-serif'] = ['SimHei']  
# plt.rcParams['axes.unicode_minus'] = False 

def cacul_y(p1_rot,p2_rot,img):
    x,y = img.shape[1],img.shape[0]
    y_list = []
    for y_index in range(y):
        if img[y_index].sum() > 0:
            y_list.append(y_index)
    min_y = min(y_list)
    max_y = max(y_list)
    p2_rot = (p2_rot[0],min_y)
    p1_rot = (p1_rot[0],max_y)
    return p1_rot,p2_rot

def rotate_contour(cnt, M):
    pts = cnt.reshape(-1, 2).astype(np.float64)
    ones = np.ones((pts.shape[0], 1))
    pts_h = np.hstack([pts, ones])   # (N, 3)
    pts_rot = pts_h @ M.T            # (N, 2)
    return pts_rot

def cacul_y_by_contour(p1_rot, p2_rot, cnt_rot):
    """
    cnt_rot: (N, 2)，旋转后的轮廓点 (x, y)
    """
    ys = cnt_rot[:, 1]
    if len(ys) == 0:
        return p1_rot, p2_rot
    min_y = int(np.min(ys))
    max_y = int(np.max(ys))

    p2_rot = (int(p2_rot[0]), min_y)
    p1_rot = (int(p1_rot[0]), max_y)
    return p1_rot, p2_rot


def cacul_x(index,img):
    x_list = []
    for x_index in range(img.shape[1]):
        if img[index,x_index] > 0:
            x_list.append(x_index)
    if x_list:  
        min_x = min(x_list)
        max_x = max(x_list)
    else:
        min_x = 0
        max_x = 0 
    return min_x,max_x

def get_axiosx(rot_img,dist):
    h,w = rot_img.shape
    i1 = 0
    for i in range(h-1,0,-1):
        min_x,max_x = cacul_x(i,rot_img)
        if max_x - min_x >= dist:
            i1 = i
            break
    return i1

# def get_axiosx(rot_img,dist):
#     h,w = rot_img.shape
#     i1 = 0
#     for i in range(h):
#         min_x,max_x = cacul_x(i,rot_img)
#         if max_x - min_x >= dist:
#             i1 = i
#             break
#     return i1

def rotate_point(pt, M):
    pt_h = np.array([pt[0], pt[1], 1.0])  
    new_pt = M @ pt_h
    return new_pt.astype(int)
    
def pil_imread(img_path, convert_to_bgr=False):
    """使用PIL读取图片，支持中文路径，默认不转换BGR（因为原代码是灰度）"""
    try:
        pil_img = Image.open(img_path)
        
        # 转换为灰度图
        if pil_img.mode != 'L':
            pil_img = pil_img.convert('L')
        
        # 转换为numpy数组
        img = np.array(pil_img)
        
        # 确保是uint8类型
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
            
        return img
    except Exception as e:
        raise ValueError(f"PIL读取图片失败: {e}")

def pil_imwrite(img_path, img):
    """使用PIL保存图片，支持中文路径"""
    try:
        if len(img.shape) == 3 and img.shape[2] == 3:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
        elif len(img.shape) == 2:
            pil_img = Image.fromarray(img)
        else:
            pil_img = Image.fromarray(img)
        ext = os.path.splitext(img_path)[1].lower()
        if ext in ['.jpg', '.jpeg']:
            pil_img.save(img_path, 'JPEG', quality=95)
        elif ext == '.png':
            pil_img.save(img_path, 'PNG')
        else:
            pil_img.save(img_path)
        return True
    except Exception as e:
        print(f"PIL保存图片失败: {e}")
        try:
            ext = os.path.splitext(img_path)[1].lower()
            if ext == '.jpg' or ext == '.jpeg':
                success, encoded_img = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            elif ext == '.png':
                success, encoded_img = cv2.imencode('.png', img)
            else:
                success, encoded_img = cv2.imencode('.png', img)
            
            if success:
                with open(img_path, 'wb') as f:
                    f.write(encoded_img.tobytes())
                return True
        except Exception as e2:
            print(f"OpenCV保存也失败: {e2}")
        
        return False

def extract_tooth_long_axis(
    img,
    output_path,
    filename,
    bin_thresh=30,
    kernel_size=5,
    axis_len_scale=1.2
):  

    # img = pil_imread(img_path)  
    h, w = img.shape
    _, mask = cv2.threshold(img, bin_thresh, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        # print(f"{filename}未检测到牙体轮廓")
        return
    tooth_cnt = max(contours, key=cv2.contourArea)
    pts = tooth_cnt.reshape(-1, 2).astype(np.float64)
    center = pts.mean(axis=0)
    cov = np.cov((pts - center).T)
    eigvals, eigvecs = np.linalg.eig(cov)
    axis = eigvecs[:, np.argmax(eigvals)]
    axis = axis / np.linalg.norm(axis)
    L = max(h, w) * axis_len_scale
    p1 = (center - axis * L).astype(int)
    p2 = (center + axis * L).astype(int)
    angle_axis = np.degrees(np.arctan2(axis[1], axis[0]))
    rotate_angle = angle_axis + 90.0
    (h, w) = img.shape[:2]
    center_pt = (int(center[0]), int(center[1]))
    M = cv2.getRotationMatrix2D(center_pt, rotate_angle, 1.0)
    rot_img = cv2.warpAffine(img, M, (w, h))
    p1_rot = rotate_point(p1, M)
    p2_rot = rotate_point(p2, M)
    tooth_cnt = rotate_contour(tooth_cnt, M)
    # p1_rot,p2_rot = cacul_y(p1_rot,p2_rot,rot_img)
    p1_rot, p2_rot = cacul_y_by_contour(p1_rot,p2_rot,tooth_cnt)
    center_rot = rotate_point(center, M)
    vis = cv2.cvtColor(rot_img, cv2.COLOR_GRAY2BGR)
    cv2.line(vis, tuple(p1_rot), tuple(p2_rot), (0, 0, 255), 3)
    cv2.circle(vis, tuple(center_rot), 5, (255, 0, 0), -1)
    # print(filename,p1_rot,p2_rot,center_rot)
    # if not pil_imwrite(output_path, vis):
    #     raise ValueError("图片保存失败")
    # cv2.imshow("vis",vis)
    # cv2.waitKey(3000)
    return rot_img,p1_rot,p2_rot,center_rot,vis

  