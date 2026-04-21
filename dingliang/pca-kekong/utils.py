import math
import os

import cv2
import numpy as np
from PIL import Image


PIXEL_SPACING_MM = 0.3


def find_first_width_perp_line(rot_img, p1_rot, p2_rot, end_index, target_width_px):
    """从图像顶部向下搜索，优先返回第一条长度大于等于目标值的垂线；若没有，则返回最接近目标值的垂线。"""
    best_line = None
    best_diff = None

    for index in range(0, min(end_index, rot_img.shape[0] - 1) + 1):
    # for index in range(0, max(0, end_index - 1)):    
        line = get_perp_line_at_index(rot_img, p1_rot, p2_rot, index)
        if line is None:
            continue

        diff = abs(line["length_px"] - target_width_px)
        if best_diff is None or diff < best_diff:
            best_diff = diff
            best_line = line

        if line["length_px"] > target_width_px:
            # print('line["length_px"]:',line["length_px"])
            return line
        # if diff <= 1.0:  # 如果长度已经非常接近目标值，就直接返回
        #     return line
    
    return best_line




# def find_first_width_perp_line(rot_img, p1_rot, p2_rot, end_index, target_width_px):
#     """从图像顶部向下搜索，优先返回第一条长度大于等于目标值的垂线；若没有，则返回最接近目标值的垂线。"""
#     for index in range(0, rot_img.shape[0]):
#         line = get_perp_line_at_index(rot_img, p1_rot, p2_rot, index)
#         if line is None:
#             continue

#         if line["length_px"] >= target_width_px:
#             print('line["length_px"]:',line["length_px"])
#             return line
    
    # return best_line



def get_blue_dashed_line(rot_img, p1_rot, p2_rot, blue_points):
    """返回经过最高蓝色像素点、垂直于长轴的蓝色虚线。"""
    if len(blue_points) == 0:
        return None

    top_idx = int(np.argmin(blue_points[:, 1]))
    top_point = tuple(map(int, blue_points[top_idx]))
    return get_perp_line_through_point(rot_img, p1_rot, p2_rot, top_point)



def cacul_xy(p1_rot, p2_rot, img, offset = 15):
    h, w = img.shape[0], img.shape[1]
    x1, y1 = p1_rot
    x2, y2 = p2_rot
    dx = x2 - x1
    dy = y2 - y1
    axis_vec = np.array([dx, dy], dtype=float)
    axis_len = np.linalg.norm(axis_vec)
    if axis_len == 0:
        return p1_rot, p2_rot

    axis_unit = axis_vec / axis_len
    points = []
    if abs(dx) < 1e-6:
        x = int(round(x1))
        if 0 <= x < w:
            for y in range(h):
                if img[y, x] > 0:
                    points.append((x, y))
    else:
        k = dy / dx
        b = y1 - k * x1
        for x in range(w):
            y = int(round(k * x + b))
            if 0 <= y < h and img[y, x] > 0:
                points.append((x, y))

    if len(points) != 0:
        p1_arr = np.array(p1_rot, dtype=float)
        proj_points = []
        for pt in points:
            proj = np.dot(np.array(pt, dtype=float) - p1_arr, axis_unit)
            proj_points.append((proj, pt))
        proj_points.sort(key=lambda x: x[0])
        p1_rot = proj_points[0][1]
        p2_rot = proj_points[-1][1]
    else:
        p1_rot, p2_rot = cacul_y(p1_rot, p2_rot, img)

    p1_arr = np.array(p1_rot, dtype=float)
    p2_arr = np.array(p2_rot, dtype=float)
    p1_off = p1_arr - axis_unit * offset
    p2_off = p2_arr + axis_unit * offset

    return (int(round(p1_off[0])), int(round(p1_off[1]))), (int(round(p2_off[0])), int(round(p2_off[1])))


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
    offset = 5
    x1 = closest_point[0] + offset
    y1 = int(round(k_perp * x1 + b_perp))
    x2 = closest_point[0] - offset
    y2 = int(round(k_perp * x2 + b_perp))
    return closest_point, min_y_point, max_y_point, k_perp
    # return closest_point, (x1, y1), (x2, y2), k_perp




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
        if 0 <= y < h and img[int(y), int(x)] > 0:
            points1.append((x, y))
    points_sorted1 = sorted(points1, key=lambda p: p[1])
    min_y_point = points_sorted1[0]
    max_y_point = points_sorted1[-1]

    offset = 20
    x1 = closest_point[0] + offset
    y1 = int(round(k_perp * x1 + b_perp))
    x2 = closest_point[0] - offset
    y2 = int(round(k_perp * x2 + b_perp))
    # return closest_point, (x1, y1), (x2, y2), k_perp
    return closest_point, min_y_point, max_y_point, k_perp


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
    # angle_axis = np.degrees(np.arctan2(axis[1], axis[0]))
    # rotate_angle = angle_axis + 90.0
    center_pt = (int(center[0]), int(center[1]))
    rotate_angle = 0
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
            lineType=cv2.LINE_AA,
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
    # print('index:',index,'length:',length)
    if length <= 0:
        return None

    return {
        "index": int(index),
        "closest_point": tuple(map(int, closest_point)),
        "start": tuple(map(int, min_y_point)),
        "end": tuple(map(int, max_y_point)),
        "length_px": float(length),
    }


def get_perp_line_through_point(rot_img, p1_rot, p2_rot, point):
    # 经过给定点、垂直于长轴的线段
    x1, y1 = p1_rot
    x2, y2 = p2_rot
    dx = x2 - x1
    dy = y2 - y1
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return None

    points = []
    h, w = rot_img.shape

    if abs(dx) < 1e-6:
        y = int(round(point[1]))
        if 0 <= y < h:
            for x in range(w):
                if rot_img[y, x] > 0:
                    points.append((x, y))
    elif abs(dy) < 1e-6:
        x = int(round(point[0]))
        if 0 <= x < w:
            for y in range(h):
                if rot_img[y, x] > 0:
                    points.append((x, y))
    else:
        k = dy / dx
        k_perp = -1.0 / k
        b_perp = point[1] - k_perp * point[0]
        if abs(k_perp) <= 1.0:
            for x in range(w):
                y = int(round(k_perp * x + b_perp))
                if 0 <= y < h and rot_img[y, x] > 0:
                    points.append((x, y))
        else:
            for y in range(h):
                x = int(round((y - b_perp) / k_perp))
                if 0 <= x < w and rot_img[y, x] > 0:
                    points.append((x, y))

    if not points:
        return None

    start = points[0]
    end = points[-1]
    length = np.linalg.norm(np.array(end) - np.array(start))
    return {
        "index": int(point[1]),
        "closest_point": tuple(map(int, point)),
        "start": tuple(map(int, start)),
        "end": tuple(map(int, end)),
        "length_px": float(length),
    }


def get_perp_line_by_axis_offset(rot_img, p1_rot, p2_rot, point, offset_px):
    # 基于蓝色虚线的经过点，沿长轴方向向上平移指定像素后构造垂线
    x1, y1 = p1_rot
    x2, y2 = p2_rot
    axis_vec = np.array([x2 - x1, y2 - y1], dtype=float)
    axis_len = np.linalg.norm(axis_vec)
    if axis_len == 0:
        return None

    axis_unit = axis_vec / axis_len
    if axis_unit[1] > 0:
        axis_unit = -axis_unit

    shifted_point = np.array(point, dtype=float) + axis_unit * float(offset_px)
    shifted_point = tuple(np.round(shifted_point).astype(int))
    return get_perp_line_through_point(rot_img, p1_rot, p2_rot, shifted_point)





def get_index_perp_line(rot_img, p1_rot, p2_rot, index, blue_points, max_search=200):
    """
    从 index 开始向上搜索，找到与 blue_points 交点数 < 4 的垂线，
    并确保再往上两条线交点 < 2 才返回
    
    Args:
        rot_img: 旋转后的图
        p1_rot, p2_rot: 长轴两点
        index: 起始 x
        blue_points: (N,2) 的 (x,y)
        max_search: 最多向上搜索多少像素
        
    Returns:
        dict or None
    """
    h = rot_img.shape[0]
    blue_points = blue_points.astype(int)
    blue_set = set((int(x), int(y)) for x, y in blue_points)

    for offset in range(max_search):
        cur_index = index - offset
        if cur_index < 0:
            break

        try:
            closest_point, min_y_point, max_y_point, _ = cacul_y_by_x(
                p1_rot, p2_rot, rot_img, cur_index
            )
        except Exception:
            continue

        length = np.linalg.norm(np.array(max_y_point) - np.array(min_y_point))
        if length <= 0:
            continue

        x0, y0 = map(int, min_y_point)
        x1, y1 = map(int, max_y_point)
        num = int(length)
        xs = np.linspace(x0, x1, num).astype(int)
        ys = np.linspace(y0, y1, num).astype(int)

        intersect_count = 0
        for x, y in zip(xs, ys):
            if (x, y) in blue_set:
                intersect_count += 1
                if intersect_count >= 5:  # 大于等于5就不用算了
                    break

        # 主条件：交点小于4
        if intersect_count < 4:
            # 再往上检查2条线
            meets_condition = True
            for next_offset in range(1, 3):  # +1, +2
                next_index = cur_index - next_offset
                if next_index < 0:
                    break
                try:
                    _, min_y_point_n, max_y_point_n, _ = cacul_y_by_x(
                        p1_rot, p2_rot, rot_img, next_index
                    )
                except Exception:
                    continue

                length_n = np.linalg.norm(np.array(max_y_point_n) - np.array(min_y_point_n))
                if length_n <= 0:
                    continue

                x0_n, y0_n = map(int, min_y_point_n)
                x1_n, y1_n = map(int, max_y_point_n)
                num_n = int(length_n)
                xs_n = np.linspace(x0_n, x1_n, num_n).astype(int)
                ys_n = np.linspace(y0_n, y1_n, num_n).astype(int)

                intersect_count_n = sum((x, y) in blue_set for x, y in zip(xs_n, ys_n))
                if intersect_count_n >= 2:
                    meets_condition = False
                    break

            if meets_condition:
                return {
                    "index": int(cur_index),
                    "closest_point": tuple(map(int, closest_point)),
                    "start": (x0, y0),
                    "end": (x1, y1),
                    "length_px": float(length),
                    "intersect_count": intersect_count,
                }

    return None


# def find_best_width_perp_line(rot_img, p1_rot, p2_rot, start_index, target_width_px):
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

    h = rot_img.shape[0]

    for index in range(start_index - 1, 0, -1):  # 注意从1开始，避免 index-1 越界

        curr_has_mask = rot_img[index].sum() > 0
        prev_has_mask = rot_img[index - 1].sum() > 0

        # ⭐ 找到上边界：从无 → 有
        if curr_has_mask and not prev_has_mask:
            line = get_perp_line_at_index1(rot_img, p1_rot, p2_rot, index)
            return line

    return None


def find_upper_tangent_perp_line(rot_img, p1_rot, p2_rot, start_index):
    for index in range(start_index - 1, 0, -1):
        line = get_perp_line_at_index1(rot_img, p1_rot, p2_rot, index)
        if line is None:
            continue
        x1, y1 = line["start"]
        x2, y2 = line["end"]
        point_count = 0

        if x1 == x2:
            ys = range(min(y1, y2), max(y1, y2) + 1)
            for y in ys:
                if 0 <= y < rot_img.shape[0] and 0 <= x1 < rot_img.shape[1] and rot_img[y, x1] > 0:
                    point_count += 1
        else:
            num_steps = int(max(abs(x2 - x1), abs(y2 - y1))) + 1
            xs = np.linspace(x1, x2, num_steps)
            ys = np.linspace(y1, y2, num_steps)
            visited = set()
            for x, y in zip(xs, ys):
                xi = int(round(x))
                yi = int(round(y))
                if (xi, yi) in visited:
                    continue
                visited.add((xi, yi))
                if 0 <= yi < rot_img.shape[0] and 0 <= xi < rot_img.shape[1] and rot_img[yi, xi] > 0:
                    point_count += 1

        if point_count <= 1:
            return line

    return None



def find_upper_tangent_perp_line(rot_img, p1_rot, p2_rot, start_index):
    for index in range(start_index - 1, 0, -1):
        line = get_perp_line_at_index1(rot_img, p1_rot, p2_rot, index)
        if line is None:
            continue

        x1, y1 = line["start"]
        x2, y2 = line["end"]
        point_count = 0

        if x1 == x2:
            ys = range(min(y1, y2), max(y1, y2) + 1)
            for y in ys:
                if 0 <= y < rot_img.shape[0] and 0 <= x1 < rot_img.shape[1] and rot_img[y, x1] > 0:
                    point_count += 1
        else:
            num_steps = int(max(abs(x2 - x1), abs(y2 - y1))) + 1
            xs = np.linspace(x1, x2, num_steps)
            ys = np.linspace(y1, y2, num_steps)
            visited = set()
            for x, y in zip(xs, ys):
                xi = int(round(x))
                yi = int(round(y))
                if (xi, yi) in visited:
                    continue
                visited.add((xi, yi))
                if 0 <= yi < rot_img.shape[0] and 0 <= xi < rot_img.shape[1] and rot_img[yi, xi] > 0:
                    point_count += 1

        if point_count <= 5:
            return line

    return None


def draw_line_by_style(img, line_info, color, dashed=False, thickness=2):
    # 按实线或虚线样式绘制一条垂线
    if line_info is None:
        return

    if dashed:
        draw_dashed_line(img, line_info["start"], line_info["end"], color, thickness)
    else:
        cv2.line(
            img,
            line_info["start"],
            line_info["end"],
            color,
            thickness,
            lineType=cv2.LINE_AA,
        )
