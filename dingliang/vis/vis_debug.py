import numpy as np
import nibabel as nib
import open3d as o3d
import os
import sys
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from nibabel.affines import apply_affine
from skimage.measure import marching_cubes

# _THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# _GET_JIETU_UTILS_DIR = os.path.normpath(
#     os.path.join(_THIS_DIR, "..", "WuYaHe-Task", "dingliang", "get_jietu")
# )
# if _GET_JIETU_UTILS_DIR not in sys.path:
#     sys.path.insert(0, _GET_JIETU_UTILS_DIR)

from utils import *
from skimage.transform import rescale
from scipy.ndimage import zoom
import numpy as  np
# Configure plotting font settings.
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
from matplotlib.colors import LinearSegmentedColormap

def has_mandibular_canal(mask):
    return np.any((mask == 2) | (mask == 3))


def save_colored_mesh_ply_ascii(
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: np.ndarray,
    out_path: str,
):
    if len(vertices) == 0 or len(faces) == 0:
        raise ValueError("网格为空，无法保存 PLY。")
    if vertices.shape[0] != colors.shape[0]:
        raise ValueError("vertices 与 colors 点数不一致。")

    if colors.dtype != np.uint8:
        colors_u8 = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
    else:
        colors_u8 = colors

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {vertices.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write(f"element face {faces.shape[0]}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for v, c in zip(vertices, colors_u8):
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")
        for tri in faces:
            f.write(f"3 {int(tri[0])} {int(tri[1])} {int(tri[2])}\n")


def process_single_ct(ct_path, base_dir, output_base_dir,txt_path):
    """
    Process one CT case end-to-end.
    """
    filename = os.path.basename(ct_path)
    filename = os.path.splitext(filename)[0]
    filename = os.path.splitext(filename)[0]
    label_path = os.path.join(base_dir, "label", f"{filename}.nii.gz")
    if not os.path.exists(label_path):
        print(f"[Skip] {filename}: label file not found - {label_path}")
        return False
    output_dir = os.path.join(output_base_dir, "screenshot", 'qianya', filename)
    output_dir1 = os.path.join(output_base_dir, "screenshot", 'houya', filename)
    # os.makedirs(output_dir1, exist_ok=True)
    # os.makedirs(output_dir, exist_ok=True)
    output_files_dir = os.path.join(base_dir, "output", filename)
    # os.makedirs(output_files_dir, exist_ok=True)

    ct_img = nib.load(ct_path)
    ct_data = ct_img.get_fdata()
    label_img = nib.load(label_path)
    label_data = label_img.get_fdata()
    # spacing_x = ct_img.get_zooms()[0]

    # Load labels and resample when spacing differs.
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

    lmax_midpoint = find_min_dist_poin(lmax_point,yagong_filt)
    lkq_midpoint = find_min_dist_poin(left_kekong,yagong_filt)

    rmax_midpoint = find_min_dist_poin(rmax_point,yagong_filt)
    rkq_midpoint = find_min_dist_poin(right_kekong,yagong_filt)

    fipoin_list = [lmax_midpoint, lkq_midpoint, rmax_midpoint, rkq_midpoint]
    qianya_list = []
    houya_list = []
    yagong_filt = np.asarray(yagong_filt)
    centers = np.asarray(fipoin_list)
    mask = np.ones(len(yagong_filt), dtype=bool)
    sorted_indices = np.argsort(yagong_filt[:, 0])  # sort by x axis
    yagong_filt_x = yagong_filt[sorted_indices]
    yagong_filt_xl,yagong_filt_xr = split_along_x(yagong_filt_x)

    if lmax_midpoint[0] > lkq_midpoint[0]:
       inde_lsta, inde_lend = get_star_end(lmax_midpoint, yagong_filt_xl, lkq_midpoint)
    else:
        inde_lsta, inde_lend = get_star_end(lkq_midpoint, yagong_filt_xl, lmax_midpoint)

    maskl = np.ones(len(yagong_filt_xl), dtype=bool)
    kekong_maskl = (yagong_filt_xl[:, 0] > yagong_filt_xl[inde_lsta, 0]) & (yagong_filt_xl[:, 0] < yagong_filt_xl[inde_lend, 0])
    maskl = maskl & ~kekong_maskl
    kekong_points_l = yagong_filt_xl[kekong_maskl]
    yagong_filt_xl = yagong_filt_xl[maskl]

    if rmax_midpoint[0] > rkq_midpoint[0]:
       inde_rsta, inde_rend = get_star_end(rmax_midpoint, yagong_filt_xr, rkq_midpoint)
    else:
        inde_rsta, inde_rend = get_star_end(rkq_midpoint, yagong_filt_xr, rmax_midpoint)
    maskr = np.ones(len(yagong_filt_xr), dtype=bool)
    kekong_maskr = (yagong_filt_xr[:, 0] > yagong_filt_xr[inde_rsta, 0]) & (yagong_filt_xr[:, 0] < yagong_filt_xr[inde_rend, 0])
    maskr = maskr & ~kekong_maskr
    kekong_points_r = yagong_filt_xr[kekong_maskr]
    yagong_filt_xr = yagong_filt_xr[maskr]

    if lmax_midpoint[0] > lkq_midpoint[0]:
        qianya_maskl = yagong_filt_xl[:,1] > lmax_midpoint[1]
        houya_maskl = yagong_filt_xl[:, 1] < lmax_midpoint[1]
    else:
        qianya_maskl = yagong_filt_xl[:,1]  > lkq_midpoint[1]
        houya_maskl = yagong_filt_xl[:, 1]  < lkq_midpoint[1]

    if rmax_midpoint[0] > rkq_midpoint[0]:
        qianya_maskr = yagong_filt_xr[:,1] < rmax_midpoint[1]
        houya_maskr = yagong_filt_xr[:,1] > rmax_midpoint[1]
    else:
        qianya_maskr = yagong_filt_xr[:,1] < rkq_midpoint[1]
        houya_maskr = yagong_filt_xr[:, 1] > rkq_midpoint[1]
    # Split and filter points by left/right masks
    houya_points_r = yagong_filt_xr[houya_maskr]
    houya_points_l = yagong_filt_xl[houya_maskl]
    qianya_points_r = yagong_filt_xr[qianya_maskr]
    qianya_points_l = yagong_filt_xl[qianya_maskl]
    qianya_points = np.vstack([houya_points_l, qianya_points_r])
    houya_points = np.vstack([qianya_points_l, houya_points_r])

    kekong_points_l_move_to_qianya = []
    kekong_points_r_move_to_qianya = []
    qianya_data = []
    houya_data = []
    kekong_data = []

    qianya_path = os.path.join(txt_path,'pca-qianya',filename,'len.txt')
    houya_path = os.path.join(txt_path, 'pca-houya', filename, 'len.txt')
    kekong_path = os.path.join(txt_path, 'pca-kekong', filename, 'len.txt')
    with open(qianya_path, 'r')as f:
        for line in f:
            line = line.strip()
            value = float(line.split(',')[-3]) 
            qianya_data.append(value)

    with open(houya_path, 'r')as f:
        for line in f:
            line = line.strip()
            # value = float(line.split(',')[-1])
            value = float(line.split(',')[-2]) 
            houya_data.append(value )

    with open(kekong_path, 'r')as f:
        for line in f:
            line = line.strip()
            # value = float(line.split(',')[-1])
            value = float(line.split(',')[-1]) 
            kekong_data.append(value )


    counter = 0
    xhg_points = np.asarray(xhg_points)
    N = len(xhg_points)
    colors_all = np.ones((N, 3)) * 0.7  # 榛樿鐏拌壊


    prev_plane = None
    prev_color = None
    # Match qianya_points processing with get_jietu/main_debug.py
    qianya_points = filt_curve(qianya_points, 0.2)
    houya_points = filt_curve(houya_points, 0.2)
    kekong_points_l = filt_curve(kekong_points_l, 0.2)
    kekong_points_r = filt_curve(kekong_points_r, 0.2)

    # kekong_points = np.vstack([kekong_points_l, kekong_points_r])
    # kekong_points = filt_curve(kekong_points, 0.2)
    # kekong_pcd = o3d.geometry.PointCloud()
    # kekong_pcd.points = o3d.utility.Vector3dVector(kekong_points)
    # houya_pcd = o3d.geometry.PointCloud()
    # houya_pcd.points = o3d.utility.Vector3dVector(houya_points)
    # vis([kekong_pcd,houya_pcd], "kekong_points")

    hy_max = np.max(houya_data)
    hy_min = np.min(houya_data)

    qy_max = np.max(qianya_data)
    qy_min = np.min(qianya_data)

    # Posterior-region visualization
    for index, point in enumerate(qianya_points):
        plane_coeffs, _ = compute_shortest_distance_to_curve_with_perpendicular_plane(
            xiayuanxiang_path_data, point
        )
        pm_len = houya_data[index]
        # print(f"point {index} bone length: {pm_len}")
        # curr_color = map_houya_length_to_color(pm_len, hy_min, hy_max)
        curr_color = get_clo_list_houya(pm_len,hy_min,hy_max)

        a, b, c, d = plane_coeffs
        n = np.array([a, b, c])
        dist = np.abs(xhg_points @ n + d) / np.linalg.norm(n)
        mask = dist < 0.3

        colors_all[mask] = curr_color
        if prev_plane is not None:
            between_mask, blended = paint_between_planes(
                xhg_points,
                prev_plane,
                plane_coeffs,
                prev_color,
                curr_color
            )
            if blended is not None:
                colors_all[between_mask] = blended
        prev_plane = plane_coeffs
        prev_color = curr_color

    if len(kekong_points_l) > 0:
        kekong_points_l = filt_curve(kekong_points_l, 0.2)
        for index, point in enumerate(kekong_points_l):
            plane_coeffs, closest_point = compute_shortest_distance_to_curve_with_perpendicular_plane(
             xiayuanxiang_path_data, point
            )
            plane_coeffs, closest_point = trans_to_nifti(
            plane_coeffs, closest_point, ct_img, affine)

            mid_slice,mid_mk = extract_slice(
            ct_data, fdata, plane_coeffs, closest_point,index, len(kekong_points_l))
            # if np.all(mid_mk == 0):
            #     kekong_points_l_move_to_qianya.append(point)
            if not has_mandibular_canal(mid_mk):
                kekong_points_l_move_to_qianya.append(point)
                # kekong_points_l.remove(point)
                kekong_points_l = kekong_points_l[~np.all(kekong_points_l == point, axis=1)]

    if len(kekong_points_r) > 0:
        kekong_points_r = filt_curve(kekong_points_r, 0.2)
        for index, point in enumerate(kekong_points_r):
            plane_coeffs, closest_point = compute_shortest_distance_to_curve_with_perpendicular_plane(
             xiayuanxiang_path_data, point
            )
            plane_coeffs, closest_point = trans_to_nifti(
            plane_coeffs, closest_point, ct_img, affine)

            mid_slice,mid_mk = extract_slice(
            ct_data, fdata, plane_coeffs, closest_point,index, len(kekong_points_r))
            # if np.all(mid_mk == 0):
            #     kekong_points_r_move_to_qianya.append(point)
                
            if not has_mandibular_canal(mid_mk):
                kekong_points_r_move_to_qianya.append(point)
                # kekong_points_r.remove(point)
                kekong_points_r = kekong_points_r[~np.all(kekong_points_r == point, axis=1)]


    kekong_points = np.vstack([kekong_points_l, kekong_points_r])

    # Canal-region visualization
    print("kekong_points:", len(kekong_points), len(kekong_data))
    # for index, point in enumerate(kekong_points):
    for index, point in zip(range(len(kekong_points) - 1), kekong_points):
        plane_coeffs, closest_point = compute_shortest_distance_to_curve_with_perpendicular_plane(
            xiayuanxiang_path_data, point
        )
        pm_len = kekong_data[index]
        # print(f"point {index} bone length: {pm_len}")
        if pm_len == 0:
            curr_color = np.array([0.7, 0.7, 0.7])  
        else:
            curr_color = get_clo_list_houya(pm_len,hy_min,hy_max)
        a, b, c, d = plane_coeffs
        n = np.array([a, b, c])
        dist = np.abs(xhg_points @ n + d) / np.linalg.norm(n)
        mask = dist < 0.3

        colors_all[mask] = curr_color
        if prev_plane is not None:
            between_mask, blended = paint_between_planes(
                xhg_points,
                prev_plane,
                plane_coeffs,
                prev_color,
                curr_color
            )
            if blended is not None:
                colors_all[between_mask] = blended
        prev_plane = plane_coeffs
        prev_color = curr_color


    if len(kekong_points_l_move_to_qianya) > 0:
        houya_points = np.vstack([np.asarray(kekong_points_l_move_to_qianya), houya_points])
    if len(kekong_points_r_move_to_qianya) > 0:
        houya_points = np.vstack([houya_points, np.asarray(kekong_points_r_move_to_qianya)])

    
    # Anterior-region visualization
    for index, point in enumerate(houya_points):
        plane_coeffs, _ = compute_shortest_distance_to_curve_with_perpendicular_plane(
            xiayuanxiang_path_data, point
        )
        pm_len = qianya_data[index]
        # print(f"point {index} bone length: {pm_len}")
        # curr_color = map_houya_length_to_color(pm_len, hy_min, hy_max)
        curr_color = get_clo_list_qianya(pm_len,qy_min,qy_max)

        a, b, c, d = plane_coeffs
        n = np.array([a, b, c])
        dist = np.abs(xhg_points @ n + d) / np.linalg.norm(n)
        mask = dist < 0.3

        colors_all[mask] = curr_color
        if prev_plane is not None:
            between_mask, blended = paint_between_planes(
                xhg_points,
                prev_plane,
                plane_coeffs,
                prev_color,
                curr_color
            )
            if blended is not None:
                colors_all[between_mask] = blended
        prev_plane = plane_coeffs
        prev_color = curr_color


    gray_color = np.array([0.7, 0.7, 0.7])
    gray_threshold = min(left_z_centroid[1], right_z_centroid[1])
    mask_gray = xhg_points[:, 1] < gray_threshold
    # Force below-threshold region to gray
    colors_all[mask_gray] = gray_color
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xhg_points)
    pcd.colors = o3d.utility.Vector3dVector(colors_all)
    # o3d.visualization.draw_geometries([pcd], width=800, height=600)

    # 将 label(>0) 重建为网格，并把当前点云颜色映射到网格顶点
    vol = (fdata > 0).astype(np.uint8)
    verts_ijk, faces, _, _ = marching_cubes(vol, level=0.5)
    verts_xyz = apply_affine(affine, verts_ijk + 0.5)


    tree = KDTree(xhg_points)
    _, nn_idx = tree.query(verts_xyz, k=1)
    mesh_colors = colors_all[nn_idx]
    verts_xyz[:, 1] *= -1.0
    verts_xyz[:, 0] *= -1.0

    ply_path = os.path.join(txt_path, "vis", f"{filename}.ply")
    save_colored_mesh_ply_ascii(verts_xyz, faces, mesh_colors, ply_path)
    print(f" {filename} 网格PLY已保存: {ply_path}")




if __name__ == "__main__":
    base_dir = r"C:\yuechen\code\wuyahe\1.code\2.data-缩放"
    txt_path = r'C:\yuechen\code\wuyahe\1.code\2.data-缩放\screenshot\pca'
    output_base_dir = base_dir
    spacing = 0.3
    ct_dir = os.path.join(base_dir, "ct")
    if not os.path.exists(ct_dir):
        print(f"[Error] CT directory not found: {ct_dir}")
        exit(1)
    ct_files = []
    for file in os.listdir(ct_dir):
        if file.endswith('.nii.gz'):
            ct_files.append(os.path.join(ct_dir, file))
    success_count = 0
    for i, ct_path in enumerate(ct_files, 1):
        # print(f"\n[{i}/{len(ct_files)}] 澶勭悊杩涘害")
        print(f"\n[{i}/{len(ct_files)}] ")
        success = process_single_ct(ct_path, base_dir, output_base_dir, txt_path)
