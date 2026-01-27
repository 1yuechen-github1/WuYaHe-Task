import math

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
  
# 设置中文字体和输出路径
# plt.rcParams['font.sans-serif'] = ['SimHei']  
# plt.rcParams['axes.unicode_minus'] = False 

def cacul_xy(p1_rot,p2_rot,img):
    h, w = img.shape[0], img.shape[1]
    # y = kx + b
    x1, y1 = p1_rot
    x2, y2 = p2_rot    
    k = (y2 - y1) / (x2 - x1)
    b = y1 - k * x1
    points = []
    for x in range(w):
        y = k * x + b
        y = int(round(y))
        if 0 <= y < h and img[int(y), int(x)] > 0:
            points.append((x, y))  
    points_sorted = sorted(points, key=lambda p: p[1])
    # print(len(points_sorted))
    if len(points_sorted) != 0:
        p1_rot = points_sorted[0]  
        p2_rot = points_sorted[-1]   
    else:
        p1_rot,p2_rot = cacul_y(p1_rot,p2_rot,img)    
    return p1_rot,p2_rot

def cacul_y_by_x(p1_rot,p2_rot,img, index):
    h, w = img.shape[0], img.shape[1]
    # y = kx + b
    x1, y1 = p1_rot
    x2, y2 = p2_rot
    k = (y2 - y1) / (x2 - x1)
    b = y1 - k * x1
    points = []
    for x in range(w):
        y = k * x + b
        y = int(round(y))
        if 0 <= y < h and img[int(y), int(x)] > 0:
            points.append((x, y))
    points_sorted = sorted(points, key=lambda p: p[1])
    closest_point = min(points_sorted,key=lambda p: abs(p[1] - index))

    k_perp = -1 / k
    b_perp = closest_point[1] - k_perp * closest_point[0]
    points1 = []
    for x in range(w):
        y = k_perp * x + b_perp
        y = int(round(y))
        if 0 <= y < h and img[int(y), int(x)] > 0:
            points1.append((x, y))
    points_sorted1 = sorted(points1, key=lambda p: p[1])
    min_y_point = points_sorted1[0]
    max_y_point = points_sorted1[-1]
    return closest_point,min_y_point,max_y_point,k_perp

def get_axios1(img,p1_rot,p2_rot):
    for i in range(p1_rot[1], p2_rot[1]):
        closest_point, min_y_point, max_y_point, k_perp = cacul_y_by_x(p1_rot,p2_rot,img, i)
        if math.sqrt((min_y_point[0] - max_y_point[0]) ** 2 + (min_y_point[1] - max_y_point[1]) ** 2) > 20:
            return min_y_point,max_y_point

def get_top_poin(img,p1_rot,p2_rot):
    x, y = img.shape[1], img.shape[0]
    for i in range(y):
        closest_point, min_y_point, max_y_point, k_perp = cacul_y_by_x(p1_rot, p2_rot, img, i)
        if img[min_y_point] == img[max_y_point] :
          return min_y_point,max_y_point




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
    for i in range(h-1,0,-1):
        min_x,max_x = cacul_x(i,rot_img)
        if max_x - min_x == dist:
            break
    return i    

def rotate_point(pt, M):
    pt_h = np.array([pt[0], pt[1], 1.0])  
    new_pt = M @ pt_h
    return new_pt.astype(int)

def rotate_points(points, M):
    ones = np.ones((points.shape[0], 1))
    points_h = np.hstack([points, ones])  # (N, 3)
    rotated = points_h @ M.T               # (N, 2)
    return rotated



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
        img_pil,
        output_path,
        filename,
        bin_thresh=50,
        kernel_size=5,
        axis_len_scale=1.2
):
    img = np.array(img_pil)
    blue_mask = (img[:, :, 0] == 0) & (img[:, :, 1] == 0) & (img[:, :, 2] == 255)
    blue_points = np.column_stack(np.where(blue_mask))

    if img_pil.mode != 'L':
        img_gray_pil = img_pil.convert('L')
        img = np.array(img_gray_pil)
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
        return
    tooth_cnt = max(contours, key=cv2.contourArea)
    pts = tooth_cnt.reshape(-1, 2).astype(np.float64)
    center = pts.mean(axis=0)
    if len(pts) < 3:
        print(filename, '点数量太少，主成分分析无效')
        return None
    else:
        cov = np.cov((pts - center).T)
        eigvals, eigvecs = np.linalg.eig(cov)
        axis = eigvecs[:, np.argmax(eigvals)]
        axis = axis / np.linalg.norm(axis)
        L = max(h, w) * axis_len_scale
        p1 = (center - axis * L).astype(int)
        p2 = (center + axis * L).astype(int)
        angle_axis = np.degrees(np.arctan2(axis[1], axis[0]))
        rotate_angle = angle_axis + 60.0
        (h, w) = img.shape[:2]
        center_pt = (int(center[0]), int(center[1]))

        M = cv2.getRotationMatrix2D(center_pt, rotate_angle, 1.0)
        rot_img = cv2.warpAffine(img, M, (w, h))
        p1_rot = rotate_point(p1, M)
        p2_rot = rotate_point(p2, M)

        blue_points = np.column_stack([blue_points[:, 1],  blue_points[:, 0],])

        blue_points = rotate_points(blue_points, M)
        center_rot = rotate_point(center, M)

        vis = cv2.cvtColor(rot_img, cv2.COLOR_GRAY2BGR)
        center_int = (int(center_rot[0]), int(center_rot[1]))
        if len(blue_points) != 0:
            center_mean = np.mean(blue_points, axis=0)
            center_x = int(round(center_mean[0]))
            center_y = int(round(center_mean[1]))
            center_mean = (center_x, center_y)
        else:
            center_mean = center_int
        # print(p1_rot, p2_rot)
        p1, p2 = cacul_xy(p1_rot, p2_rot, rot_img)

        return rot_img, p1, p2, center_mean, vis

  