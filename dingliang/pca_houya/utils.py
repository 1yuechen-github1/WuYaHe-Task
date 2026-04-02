import math
import os

import cv2
import numpy as np
from PIL import Image


PIXEL_SPACING_MM = 0.3


def cacul_xy(p1_rot, p2_rot, img,offset = 10):
    h, w = img.shape[0], img.shape[1]
    x1, y1 = p1_rot
    x2, y2 = p2_rot
    k = (y2 - y1) / (x2 - x1)
    b = y1 - k * x1
    points = []
    # points1 = []
    for x in range(w):
        y = k * x + b
        y = int(round(y))
        if 0 <= y < h and img[int(y), int(x)] > 0:
            points.append((x, y))
        # if 0 <= y < h:
        #     # points1.append((x, y))
    points_sorted = sorted(points, key=lambda p: p[1])
    if len(points_sorted) != 0:
        p1_rot = points_sorted[0]
        p2_rot = points_sorted[-1]
    else:
        p1_rot, p2_rot = cacul_y(p1_rot, p2_rot, img)
    
    x3 = p1_rot[0] + offset
    y3 = int(round(k * x3 + b))
    x4 = p2_rot[0] - offset
    y4 = int(round(k * x4 + b))
    
    return (x3, y3), (x4, y4)


def cacul_y_by_x(p1_rot, p2_rot, img, index):
    h, w = img.shape[0], img.shape[1]
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
    closest_point = min(points_sorted, key=lambda p: abs(p[1] - index))

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
    return closest_point, min_y_point, max_y_point, k_perp




def cacul_y_by_x1(p1_rot, p2_rot, img, index):
    # 考虑像素等于0的情况，在长轴上的指定位置构造一条垂直于长轴的截线
    h, w = img.shape[0], img.shape[1]
    x1, y1 = p1_rot
    x2, y2 = p2_rot
    k = (y2 - y1) / (x2 - x1)
    b = y1 - k * x1
    points = []
    for x in range(w):
        y = k * x + b
        y = int(round(y))
        if 0 <= y < h:
            points.append((x, y))
    points_sorted = sorted(points, key=lambda p: p[1])
    closest_point = min(points_sorted, key=lambda p: abs(p[1] - index))

    k_perp = -1 / k
    b_perp = closest_point[1] - k_perp * closest_point[0]
    points1 = []
    for x in range(w):
        y = k_perp * x + b_perp
        y = int(round(y))
        if 0 <= y < h :
            points1.append((x, y))
    points_sorted1 = sorted(points1, key=lambda p: p[1])
    min_y_point = points_sorted1[0]
    max_y_point = points_sorted1[-1]

    offset = 20
    x1 = closest_point[0] + offset
    y1 = int(round(k_perp * x1 + b_perp))
    x2 = closest_point[0] - offset
    y2 = int(round(k_perp * x2 + b_perp))
    return closest_point, (x1, y1), (x2, y2), k_perp


def get_axios1(img, p1_rot, p2_rot):
    for i in range(p1_rot[1], p2_rot[1]):
        _, min_y_point, max_y_point, _ = cacul_y_by_x(p1_rot, p2_rot, img, i)
        if math.sqrt((min_y_point[0] - max_y_point[0]) ** 2 + (min_y_point[1] - max_y_point[1]) ** 2) > 20:
            return min_y_point, max_y_point


def get_top_poin(img, p1_rot, p2_rot):
    _, h = img.shape[1], img.shape[0]
    for i in range(h):
        _, min_y_point, max_y_point, _ = cacul_y_by_x(p1_rot, p2_rot, img, i)
        if img[min_y_point] == img[max_y_point]:
            return min_y_point, max_y_point


def cacul_y(p1_rot, p2_rot, img):
    _, h = img.shape[1], img.shape[0]
    y_list = []
    for y_index in range(h):
        if img[y_index].sum() > 0:
            y_list.append(y_index)
    min_y = min(y_list)
    max_y = max(y_list)
    p2_rot = (p2_rot[0], min_y)
    p1_rot = (p1_rot[0], max_y)
    return p1_rot, p2_rot


def cacul_x(index, img):
    x_list = []
    for x_index in range(img.shape[1]):
        if img[index, x_index] > 0:
            x_list.append(x_index)
    if x_list:
        min_x = min(x_list)
        max_x = max(x_list)
    else:
        min_x = 0
        max_x = 0
    return min_x, max_x


def get_axiosx(rot_img, dist):
    h, _ = rot_img.shape
    for i in range(h - 1, 0, -1):
        min_x, max_x = cacul_x(i, rot_img)
        if max_x - min_x == dist:
            break
    return i


def rotate_point(pt, M):
    pt_h = np.array([pt[0], pt[1], 1.0])
    new_pt = M @ pt_h
    return new_pt.astype(int)


def rotate_points(points, M):
    ones = np.ones((points.shape[0], 1))
    points_h = np.hstack([points, ones])
    rotated = points_h @ M.T
    return rotated


def pil_imwrite(img_path, img):
    try:
        if len(img.shape) == 3 and img.shape[2] == 3:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
        else:
            pil_img = Image.fromarray(img)
        ext = os.path.splitext(img_path)[1].lower()
        if ext in [".jpg", ".jpeg"]:
            pil_img.save(img_path, "JPEG", quality=95)
        elif ext == ".png":
            pil_img.save(img_path, "PNG")
        else:
            pil_img.save(img_path)
        return True
    except Exception:
        return False


def extract_tooth_long_axis(
    img_pil,
    output_path,
    filename,
    bin_thresh=40,
    kernel_size=5,
    axis_len_scale=1.2,
    offset=10,
):
    img = np.array(img_pil)
    blue_mask = (img[:, :, 0] == 0) & (img[:, :, 1] == 0) & (img[:, :, 2] == 255)
    blue_points = np.column_stack(np.where(blue_mask))

    if img_pil.mode != "L":
        img_gray_pil = img_pil.convert("L")
        img = np.array(img_gray_pil)
    h, w = img.shape
    _, mask = cv2.threshold(img, bin_thresh, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return
    tooth_cnt = max(contours, key=cv2.contourArea)
    pts = tooth_cnt.reshape(-1, 2).astype(np.float64)
    center = pts.mean(axis=0)
    if len(pts) < 3:
        print(filename, "轮廓点太少，无法计算长轴")
        return None

    cov = np.cov((pts - center).T)
    eigvals, eigvecs = np.linalg.eig(cov)
    axis = eigvecs[:, np.argmax(eigvals)]
    axis = axis / np.linalg.norm(axis)
    L = max(h, w) * axis_len_scale
    p1 = (center - axis * L).astype(int)
    p2 = (center + axis * L).astype(int)
    angle_axis = np.degrees(np.arctan2(axis[1], axis[0]))
    rotate_angle = angle_axis + 60.0
    center_pt = (int(center[0]), int(center[1]))

    M = cv2.getRotationMatrix2D(center_pt, rotate_angle, 1.0)
    rot_img = cv2.warpAffine(img, M, (w, h))
    p1_rot = rotate_point(p1, M)
    p2_rot = rotate_point(p2, M)

    blue_points = np.column_stack([blue_points[:, 1], blue_points[:, 0]])
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

    p1, p2 = cacul_xy(p1_rot, p2_rot, rot_img, offset=offset)

    return rot_img, p1, p2 , center_mean, vis, blue_points


def mm_to_px(mm):
    # 按 0.3mm/像素把毫米换算成像素
    return max(1, int(round(mm / PIXEL_SPACING_MM)))


def draw_dashed_line(img, pt1, pt2, color, thickness=1, dash_length=8):
    # 按给定起点和终点绘制虚线
    pt1 = np.array(pt1, dtype=float)
    pt2 = np.array(pt2, dtype=float)
    line_vec = pt2 - pt1
    line_len = np.linalg.norm(line_vec)
    if line_len == 0:
        return

    direction = line_vec / line_len
    for start in np.arange(0, line_len, dash_length * 2):
        end = min(start + dash_length, line_len)
        seg_start = pt1 + direction * start
        seg_end = pt1 + direction * end
        cv2.line(
            img,
            tuple(np.round(seg_start).astype(int)),
            tuple(np.round(seg_end).astype(int)),
            color,
            thickness,
        )


def get_perp_line_at_index1(rot_img, p1_rot, p2_rot, index):
    # 在长轴上的指定位置构造一条垂直于长轴的截线，考虑像素等于0的情况
    if index < 0 or index >= rot_img.shape[0]:
        return None

    try:
        closest_point, min_y_point, max_y_point, _ = cacul_y_by_x1(p1_rot, p2_rot, rot_img, index)
    except Exception:
        return None

    length = np.linalg.norm(np.array(max_y_point) - np.array(min_y_point))
    if length <= 0:
        return None

    return {
        "index": int(index),
        "closest_point": tuple(map(int, closest_point)),
        "start": tuple(map(int, min_y_point)),
        "end": tuple(map(int, max_y_point)),
        "length_px": float(length),
    }

def get_perp_line_at_index(rot_img, p1_rot, p2_rot, index):
    # 在长轴上的指定位置构造一条垂直于长轴的截线
    if index < 0 or index >= rot_img.shape[0]:
        return None

    try:
        closest_point, min_y_point, max_y_point, _ = cacul_y_by_x(p1_rot, p2_rot, rot_img, index)
    except Exception:
        return None

    length = np.linalg.norm(np.array(max_y_point) - np.array(min_y_point))
    if length <= 0:
        return None

    return {
        "index": int(index),
        "closest_point": tuple(map(int, closest_point)),
        "start": tuple(map(int, min_y_point)),
        "end": tuple(map(int, max_y_point)),
        "length_px": float(length),
    }


def find_best_width_perp_line(rot_img, p1_rot, p2_rot, start_index, target_width_px):
    # 从起始位置向上搜索，找到长度最接近目标值的垂线
    best_line = None
    best_diff = None

    for index in range(start_index, -1, -1):
        line = get_perp_line_at_index(rot_img, p1_rot, p2_rot, index)
        if line is None:
            continue

        diff = abs(line["length_px"] - target_width_px)
        if best_diff is None or diff < best_diff:
            best_diff = diff
            best_line = line

    return best_line


# def find_upper_tangent_perp_line(rot_img, p1_rot, p2_rot, start_index):
    # 从绿色实线继续向上搜索，返回离开 mask 前最后一条有效垂线，
    # 用它近似表示与 mask 上边界相切的绿色虚线
    last_valid_line = None

    for index in range(start_index - 1, -1, -1):
        line = get_perp_line_at_index1(rot_img, p1_rot, p2_rot, index)
        if line is not None:
            last_valid_line = line
            continue
        # if rot_img[index].sum() > 0 and rot_img[index - 1].sum() == 0:
        #     return last_valid_line
        if last_valid_line is not None:
            return last_valid_line
    return last_valid_line


def find_upper_tangent_perp_line(rot_img, p1_rot, p2_rot, start_index):

    h = rot_img.shape[0]

    for index in range(start_index - 1, 0, -1):  # 注意从1开始，避免 index-1 越界

        curr_has_mask = rot_img[index].sum() > 0
        prev_has_mask = rot_img[index - 1].sum() > 0

        # ⭐ 找到上边界：从无 → 有
        if curr_has_mask and not prev_has_mask:
            line = get_perp_line_at_index1(rot_img, p1_rot, p2_rot, index)
            return line

    return None



def draw_line_by_style(img, line_info, color, dashed=False, thickness=2):
    # 按实线或虚线样式绘制一条垂线
    if line_info is None:
        return

    if dashed:
        draw_dashed_line(img, line_info["start"], line_info["end"], color, thickness)
    else:
        cv2.line(img, line_info["start"], line_info["end"], color, thickness)
