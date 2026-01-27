import numpy as np
import nibabel as nib
import open3d as o3d
import os
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils import *
from skimage.transform import rescale
from scipy.ndimage import zoom
import numpy as  np
# 设置中文字体和输出路径
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
from matplotlib.colors import LinearSegmentedColormap


def process_single_ct(ct_path, base_dir, output_base_dir,txt_path):
    """
    处理单个CT文件的完整流程
    """
    filename = os.path.basename(ct_path)
    filename = os.path.splitext(filename)[0]
    filename = os.path.splitext(filename)[0]
    label_path = os.path.join(base_dir, "label", f"{filename}.nii.gz")
    if not os.path.exists(label_path):
        print(f"❌ 跳过 {filename}: 对应的label文件不存在 - {label_path}")
        return False
    output_dir = os.path.join(output_base_dir, "screenshot", 'qianya', filename)
    output_dir1 = os.path.join(output_base_dir, "screenshot", 'houya', filename)
    os.makedirs(output_dir1, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    output_files_dir = os.path.join(base_dir, "output", filename)
    os.makedirs(output_files_dir, exist_ok=True)

    ct_img = nib.load(ct_path)
    ct_data = ct_img.get_fdata()
    label_img = nib.load(label_path)
    label_data = label_img.get_fdata()
    # spacing_x = ct_img.get_zooms()[0]

    # 下颌骨/左右下颌管/根据测地距离定位颏孔
    img = nib.load(label_path)
    fdata = img.get_fdata()
    affine = img.affine
    shape = fdata.shape
    header = img.header
    img_spacing = header.get_zooms()

    if img_spacing[0] != spacing or img_spacing[1] != spacing or img_spacing[2] != spacing:
        sca_fac_x = img_spacing[0] / spacing
        sca_fac_y = img_spacing[1] / spacing
        sca_fac_z = img_spacing[2] / spacing
        scale = (sca_fac_x, sca_fac_y, sca_fac_z)
        ct_data = rescale(ct_data, scale, order=3, preserve_range=True, mode='reflect', anti_aliasing=False)
        fdata = rescale(fdata, scale, order=0, preserve_range=True, mode='edge', anti_aliasing=False)
        n_affine = affine.copy()
        for i in range(3):
            n_affine[:3, i] *= spacing / img_spacing[i]
        affine = n_affine

    xhg_points = nifti_to_pointcloud(img, fdata, affine, output_dir=output_dir, label_value=1)
    left_xiaheguan_data = nifti_to_pointcloud(img, fdata, affine, output_dir=output_dir, label_value=2)
    right_xiaheguan_data = nifti_to_pointcloud(img, fdata, affine, output_dir=output_dir, label_value=3)

    left_kekong_start_idx = np.argmin(left_xiaheguan_data[:, 1])
    labels, n_components = find_connected_components(left_xiaheguan_data)
    end_idx, geo_dists = find_end_point(left_xiaheguan_data, left_kekong_start_idx, labels, n_components)
    left_kekong = left_xiaheguan_data[end_idx]
    stl_point = left_xiaheguan_data[left_kekong_start_idx]

    right_kekong_start_idx = np.argmin(right_xiaheguan_data[:, 1])
    labels, n_components = find_connected_components(right_xiaheguan_data)
    end_idx, geo_dists = find_end_point(right_xiaheguan_data, right_kekong_start_idx, labels, n_components)
    right_kekong = right_xiaheguan_data[end_idx]
    str_point = right_xiaheguan_data[right_kekong_start_idx]

    # 把nifti转化点云后提取下缘线
    # left_z_centroid = compute_y_mid_point(left_xiaheguan_data)
    # right_z_centroid = compute_y_mid_point(right_xiaheguan_data)
    points = extract_mandible_inferior_points_from_txt(
        points=xhg_points,
        slice_thickness=2.0,
        separate_sides=True,
        num_nearest=27000,
    )
    cropped = process_point_cloud_txt(
        points=points
    )
    interp_centerline = extract_and_interpolate_centerline(
        points=cropped,
        num_steps=100,
        interpolation_step=0.5,
    )
    original_points = interp_centerline
    smoothing_params = [0.1, 0.5, 1.0]
    best_result = None
    best_metrics = None
    for smoothing in smoothing_params:
        fitted_curve, tangents, tck = spline_fit_3d(
            original_points,
            smoothing=smoothing,
            degree=3
        )
        metrics = calculate_fitting_metrics(original_points, fitted_curve)
        if best_metrics is None or metrics['rmse'] < best_metrics['rmse']:
            best_metrics = metrics
            best_result = (fitted_curve, tangents, tck, smoothing)
    if best_result:
        fitted_curve, tangents, tck, best_smoothing = best_result

    xiayuanxiang_path_data = fitted_curve
    original_z = xiayuanxiang_path_data[:, 2].copy()
    max_y = find_max_y_point(xiayuanxiang_path_data)
    midline_point = (left_kekong + right_kekong) / 2
    offset_z = int(abs(midline_point[2] - max_y[2]))
    dist = np.linalg.norm(xiayuanxiang_path_data - midline_point, axis=1)

    point = xiayuanxiang_path_data[np.argmin(dist)]
    point_z = point[2]
    dist_list = []
    for step in range(offset_z):
        current_z = point_z + step
        dist1 = abs(current_z - abs((left_kekong[2] + right_kekong[2]) / 2))
        dist_list.append(dist1)
    optimal_offset = np.argmin(dist_list)
    # print(offset_z , optimal_offset, abs(point[2] - (left_kekong[2] + right_kekong[2]) / 2))
    yagong_point = xiayuanxiang_path_data.copy()
    yagong_point[:, 2] = original_z + optimal_offset * 1

    left_z_centroid = (left_kekong + stl_point) / 2
    right_z_centroid = (right_kekong + str_point) / 2
    filt_l, filt_r = left_z_centroid[0], right_z_centroid[0]
    mask = (yagong_point[:, 0] > filt_l) & (yagong_point[:, 0] < filt_r)
    yagong_filt = yagong_point[mask]
    lmax_index = np.argmax(left_xiaheguan_data[:, 1])
    rmax_index = np.argmax(right_xiaheguan_data[:, 1])
    lmax_point = left_xiaheguan_data[lmax_index]
    rmax_point = right_xiaheguan_data[rmax_index]

    lmax_midpoint = find_min_dist_poin(lmax_point, yagong_filt)
    lkq_midpoint = find_min_dist_poin(left_kekong, yagong_filt)

    rmax_midpoint = find_min_dist_poin(rmax_point, yagong_filt)
    rkq_midpoint = find_min_dist_poin(right_kekong, yagong_filt)

    fipoin_list = [lmax_midpoint, lkq_midpoint, rmax_midpoint, rkq_midpoint]
    qianya_list = []
    houya_list = []
    yagong_filt = np.asarray(yagong_filt)
    centers = np.asarray(fipoin_list)
    mask = np.ones(len(yagong_filt), dtype=bool)
    for center in centers:
        distances = np.linalg.norm(yagong_filt - center, axis=1)
        mask = mask & (distances > 2)
        filtered_points = yagong_filt[mask]
    mid_pos = (left_kekong + right_kekong) / 2
    mid_y = mid_pos[1]
    y_coords = filtered_points[:, 1]
    qianya_mask = y_coords > mid_y
    houya_mask = y_coords <= mid_y
    qianya_points = filtered_points[qianya_mask]
    houya_points = filtered_points[houya_mask]
    qianya_list.append(qianya_points)
    houya_list.append(houya_points)

    qianya_data = []
    houya_data = []
    qianya_path = os.path.join(txt_path,'qianya',filename,'len.txt')
    houya_path = os.path.join(txt_path, 'houya', filename, 'len.txt')
    with open(qianya_path, 'r')as f:
        for line in f:
            line = line.strip()
            value = int(line.split(',')[-1]) * 0.3
            qianya_data.append(value)
    with open(houya_path, 'r')as f:
        for line in f:
            line = line.strip()
            value = int(line.split(',')[-1]) * 0.3
            houya_data.append(value)
    print('qianya_data:',qianya_data)
    print('houya_data:',houya_data)
    colormap = 'winter'
    cmap = plt.get_cmap(colormap)
    cmap = truncate_colormap(cmap, 0, 1)

    qianya_arr = np.array(qianya_data)
    houya_arr = np.array(houya_data)
    q_norm = (qianya_arr - qianya_arr.min()) / (qianya_arr.max() - qianya_arr.min() + 1e-8)
    h_norm = (houya_arr - houya_arr.min()) / (houya_arr.max() - houya_arr.min() + 1e-8)

    # 映射颜色
    qianya_colors = cmap(q_norm)[:, :3]
    houya_colors = cmap(h_norm)[:, :3]
    counter = 0
    xhg_points = np.asarray(xhg_points)
    N = len(xhg_points)
    colors_all = np.ones((N, 3)) * 0.7  # 默认灰色
    print('开始输出前牙可视化')
    for index, point in enumerate(qianya_points):
        plane_coeffs, closest_point = compute_shortest_distance_to_curve_with_perpendicular_plane(
            xiayuanxiang_path_data, point
        )
        a, b, c, d = plane_coeffs
        color = qianya_colors[index]  # 每个 plane 一个颜色
        # 点到平面距离
        dist = np.abs(
            xhg_points @ np.array([a, b, c]) + d
        ) / np.linalg.norm([a, b, c])
        mask = dist < 1
        colors_all[mask] = color
    print('开始输出后牙可视化')
    for index, point in enumerate(houya_points):
        plane_coeffs, closest_point = compute_shortest_distance_to_curve_with_perpendicular_plane(
            xiayuanxiang_path_data, point
        )
        a, b, c, d = plane_coeffs
        color = houya_colors[index]  # 每个 plane 一个颜色
        # 点到平面距离
        dist = np.abs(
            xhg_points @ np.array([a, b, c]) + d
        ) / np.linalg.norm([a, b, c])
        mask = dist < 1.0
        colors_all[mask] = color

    gray_color = np.array([0.7, 0.7, 0.7])
    mask_gray = (xhg_points[:, 1] < left_z_centroid[1]) | (xhg_points[:, 1] < left_z_centroid[1])
    # 强制设置为灰色
    colors_all[mask_gray] = gray_color

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xhg_points)
    pcd.colors = o3d.utility.Vector3dVector(colors_all)
    o3d.visualization.draw_geometries([pcd])

    colors_255 = (colors_all * 255).astype(np.uint8)
    fina_xhg = np.hstack([xhg_points,colors_255])
    os.makedirs(os.path.join(txt_path, 'vis'),exist_ok=True)
    np.savetxt(os.path.join(txt_path, 'vis', f"{filename}.txt"), fina_xhg)

    print(f" {filename} 处理完成")


if __name__ == "__main__":
    base_dir = r"C:\yuechen\code\wuyahe\1.code\2.data-缩放"
    txt_path = r'C:\yuechen\code\wuyahe\1.code\2.data-缩放\screenshot\pca-sum'
    output_base_dir = base_dir
    spacing = 0.3
    ct_dir = os.path.join(base_dir, "ct")
    if not os.path.exists(ct_dir):
        print(f"❌ CT目录不存在: {ct_dir}")
        exit(1)
    ct_files = []
    for file in os.listdir(ct_dir):
        if file.endswith('.nii.gz'):
            ct_files.append(os.path.join(ct_dir, file))
    print(f"找到 {len(ct_files)} 个CT文件")
    success_count = 0
    for i, ct_path in enumerate(ct_files, 1):
        print(f"\n[{i}/{len(ct_files)}] 处理进度")
        success = process_single_ct(ct_path, base_dir, output_base_dir, txt_path)
