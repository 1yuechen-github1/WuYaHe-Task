import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os

from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import affine_transform
from scipy.ndimage import map_coordinates
import open3d as o3d
from scipy.spatial import KDTree, cKDTree
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse.csgraph import dijkstra, connected_components
from scipy.optimize import minimize
import os
from sklearn.decomposition import PCA
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy.spatial import KDTree
from pathlib import Path
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
from scipy.ndimage import label

from PIL import Image
import numpy as np
import os


def create_plane_from_three_points(p1, p2, p3):
    """
    根据三个点创建平面方程
    """
    v1 = p2 - p1
    v2 = p3 - p1
    normal = np.cross(v1, v2)
    normal_length = np.linalg.norm(normal)
    normal = normal / normal_length
    a, b, c = normal
    d = -np.dot(normal, p1)
    return np.array([a, b, c, d])


def split_along_x(points):
    """
    将点云数据沿X轴对半切分为两部分
    """
    x_mid = np.median(points[:, 0])
    left_mask = points[:, 0] <= x_mid
    right_mask = points[:, 0] > x_mid

    left_half = points[left_mask]
    right_half = points[right_mask]

    return left_half, right_half


def nifti_to_txt_pointcloud(
        input_nifti_path,
        output_txt_path=None,
        label_value=1,
        verbose=True,
        ct_data=None,
):
    """
    将NIfTI文件中指定label值的区域转换为TXT格式的点云（每行x y z）

    """
    img = nib.load(input_nifti_path)
    affine = img.affine
    label_coords = np.argwhere(ct_data == label_value)

    if len(label_coords) == 0:
        raise ValueError(f"在文件中未找到label={label_value}的体素")

    homogeneous_coords = np.c_[label_coords, np.ones(len(label_coords))]
    world_coords = (affine @ homogeneous_coords.T).T[:, :3]
    if output_txt_path is None:
        base_name = os.path.splitext(os.path.basename(input_nifti_path))[0]
        if base_name.endswith('.nii'):
            base_name = base_name[:-4]
        output_txt_path = os.path.join(
            os.path.dirname(input_nifti_path),
            f"{base_name}_step1.txt"
        )
    np.savetxt(output_txt_path, world_coords, fmt='%.6f')
    return output_txt_path


def compute_y_mid_point(points, tolerance=1e-6):
    """
    返回Y方向中点对应的XYZ坐标 [x, y_mid, z]
    """
    y_mid = (np.max(points[:, 1]) + np.min(points[:, 1])) / 2
    mask = np.abs(points[:, 1] - y_mid) <= tolerance
    filtered_points = points[mask]
    if len(filtered_points) == 0:
        closest_idx = np.argmin(np.abs(points[:, 1] - y_mid))
        return points[closest_idx]
    x_center = np.mean(filtered_points[:, 0])
    z_center = np.mean(filtered_points[:, 2])

    return np.array([x_center, y_mid, z_center])


def create_plane_perpendicular_to_line(line_direction, closest_point, midpoint):
    """
    创建垂直于给定直线的平面
    # 返回平面方程的系数 (A, B, C, D)
    """
    # 平面的法向量就是直线的方向向量
    a, b, c = line_direction
    x0, y0, z0 = closest_point
    d = -(a * x0 + b * y0 + c * z0)
    return a, b, c, d


def create_plane_perpendicular_to_line_mid(line_direction, closest_point, midpoint):
    """
    创建垂直于给定直线的平面
    # 返回平面方程的系数 (A, B, C, D)
    """
    # # 平面的法向量就是直线的方向向量
    a, b, c = line_direction
    x0, y0, z0 = midpoint
    d = -(a * x0 + b * y0 + c * z0)
    return a, b, c, d


def find_max_y_point(points):
    """
    找到点云中Y坐标最大的点
    """
    return points[np.argmax(points[:, 1])]


def compute_shortest_distance_to_curve_with_perpendicular_plane(curve_points=None, mid_point=None):
    """
    计算平面点云到曲线的最短距离，使用垂直于曲线的平面
    """
    curve_data = fit_curve(curve_points)
    # 找到曲线上距离mid_point最近的点
    tree = KDTree(curve_points)
    distances_to_mid, indices_to_mid = tree.query([mid_point])
    closest_curve_point_idx = indices_to_mid[0]
    closest_curve_point = curve_points[closest_curve_point_idx]
    # 获取导数函数
    curve_derivative = get_curve_derivative(curve_data)
    curve_direction = curve_derivative(closest_curve_point)
    plane_coeffs = create_plane_perpendicular_to_line(curve_direction, closest_curve_point, mid_point)
    return plane_coeffs, closest_curve_point


def get_plane(plane_points=None, curve_points=None, mid_point=None):
    """
    计算平面点云到曲线的最短距离，使用垂直于曲线的平面
    """
    # 拟合曲线得到曲线方程
    curve_data = fit_curve(curve_points)
    tree = KDTree(curve_points)
    distances_to_mid, indices_to_mid = tree.query([mid_point])
    closest_curve_point_idx = indices_to_mid[0]  # 这是一个整数索引
    closest_curve_point = curve_points[closest_curve_point_idx]  # 直接使用索引
    curve_derivative = get_curve_derivative(curve_data)
    curve_direction = curve_derivative(closest_curve_point)
    plane_coeffs = create_plane_perpendicular_to_line_mid(curve_direction, closest_curve_point, mid_point)
    return plane_coeffs, closest_curve_point


def fit_curve(curve_points):
    """
    拟合曲线点云得到曲线方程
    """
    # 示例：3次多项式拟合
    x = curve_points[:, 0]
    y = curve_points[:, 1]
    z = curve_points[:, 2]
    # 假设以x为参数
    poly_y = np.polyfit(x, y, 3)
    poly_z = np.polyfit(x, z, 3)
    return {'poly_y': poly_y, 'poly_z': poly_z, 'param': 'x'}


def get_curve_derivative(curve_data):
    """
    获取曲线的导数函数
    """
    if curve_data['param'] == 'x':
        poly_y_deriv = np.polyder(curve_data['poly_y'])
        poly_z_deriv = np.polyder(curve_data['poly_z'])

        def derivative(point):
            x = point[0]
            dy_dx = np.polyval(poly_y_deriv, x)
            dz_dx = np.polyval(poly_z_deriv, x)
            tangent = np.array([1.0, dy_dx, dz_dx])
            return tangent / np.linalg.norm(tangent)  # 单位化

        return derivative


def find_closest_point_on_curve(curve_data, point, curve_points):
    """
    在曲线上找到距离给定点最近的点
    """
    if curve_data['param'] == 'x':
        # 获取曲线点的x范围
        x_min = np.min(curve_points[:, 0])
        x_max = np.max(curve_points[:, 0])

        def distance_to_curve(x):
            y = np.polyval(curve_data['poly_y'], x)
            z = np.polyval(curve_data['poly_z'], x)
            curve_point = np.array([x, y, z])
            return np.linalg.norm(curve_point - point)

        result = minimize(distance_to_curve, x0=point[0], bounds=[(x_min, x_max)])
        x_opt = result.x[0]
        y_opt = np.polyval(curve_data['poly_y'], x_opt)
        z_opt = np.polyval(curve_data['poly_z'], x_opt)

        return np.array([x_opt, y_opt, z_opt])


def point_to_plane_distance(point, plane_coeffs):
    """
    计算点到平面的有符号距离
    """
    a, b, c, d = plane_coeffs
    x, y, z = point
    numerator = a * x + b * y + c * z + d  # 移除abs()
    denominator = np.sqrt(a ** 2 + b ** 2 + c ** 2)
    return numerator / denominator


def filter_points_by_perpendicular_planes(jaw_points, left_plane_coeffs, right_plane_coeffs):
    """
    计算下颌骨点到左右垂直平面的距离，保留距离大于0的点
    """
    filtered_points = []

    for point in jaw_points:
        dist_left = point_to_plane_distance(point, left_plane_coeffs)
        dist_right = point_to_plane_distance(point, right_plane_coeffs)
        if dist_left < 0 and dist_right > 0:
            filtered_points.append(point)
    return np.array(filtered_points)



def find_closest_point(curve_points, target_point):
    distances = np.linalg.norm(curve_points - target_point[:2], axis=1)
    return np.argmin(distances)


def trans_to_nifti(plane_coeffs, pointcloud_point, nifti_img, affine):
    """
    将点云坐标系中的平面转换到NIfTI坐标系
    """
    # affine = nifti_img.affine
    point_homogeneous = np.array([pointcloud_point[0], pointcloud_point[1], pointcloud_point[2], 1])
    nifti_voxel_coords = np.linalg.inv(affine) @ point_homogeneous
    nifti_center_point = nifti_voxel_coords[:3]

    A, B, C, D = plane_coeffs
    normal = np.array([A, B, C])
    rotation_scale = affine[:3, :3]
    nifti_normal = rotation_scale.T @ normal
    # 重新计算D系数
    nifti_D = -np.dot(nifti_normal, nifti_center_point)
    nifti_plane_coeffs = [nifti_normal[0], nifti_normal[1], nifti_normal[2], nifti_D]
    return nifti_plane_coeffs, nifti_center_point

def find_connected_components(points, k=15):
    """找到点云的连通分量"""
    tree = KDTree(points)
    dists, indices = tree.query(points, k=k + 1)
    adj = np.full((len(points), len(points)), np.inf)
    for i in range(len(points)):
        for j_idx, j in enumerate(indices[i]):
            if i != j:
                adj[i, j] = dists[i][j_idx]
    n_components, labels = connected_components(adj, directed=False)
    return labels, n_components


def geodesic_with_radial_constraint(points, start_idx, k=15):
    """计算带径向约束的测地距离"""
    centered = points - points.mean(axis=0)
    main_axis = np.linalg.eigh(centered.T @ centered)[1][:, 2]  # 主方向

    tree = KDTree(points)
    dists, indices = tree.query(points, k=k + 1)

    adj = np.full((len(points), len(points)), np.inf)

    for i in range(len(points)):
        for j_idx, j in enumerate(indices[i]):
            if i != j:
                vec = points[j] - points[i]
                cos_angle = np.abs(np.dot(vec, main_axis)) / np.linalg.norm(vec)
                penalty = 5.0 if cos_angle < 0.3 else 1.0
                adj[i, j] = dists[i][j_idx] * penalty

    return dijkstra(adj, indices=[start_idx], directed=False)[0]


def geodesic_in_component(points, start_idx, component_mask, k=15):
    """在指定连通分量内计算测地距离"""
    component_points = points[component_mask]
    component_indices = np.where(component_mask)[0]

    print(f"分量内点数: {len(component_points)}, 起点是否在分量内: {start_idx in component_indices}")
    if start_idx not in component_indices:
        tree = KDTree(component_points)
        start_point = points[start_idx].reshape(1, -1)
        result = tree.query(start_point, k=1)
        print(f"KDTree查询结果类型: {type(result)}, 长度: {len(result)}")
        if isinstance(result, tuple) and len(result) == 2:
            dists, indices = result
            nearest_idx = indices[0]  # 第一个元素的索引
        else:
            dists = result[0]
            indices = result[1]
            nearest_idx = indices[0]
        new_start_idx = component_indices[nearest_idx]
    else:
        # 找到起点在分量内的位置
        new_start_idx_in_comp = np.where(component_indices == start_idx)[0][0]
        new_start_idx = start_idx
    centered = component_points - component_points.mean(axis=0)
    covariance = centered.T @ centered
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    main_axis = eigenvectors[:, 2]  # 主方向

    tree_component = KDTree(component_points)
    k_val = min(k + 1, len(component_points))
    dists, indices = tree_component.query(component_points, k=k_val)

    adj = np.full((len(component_points), len(component_points)), np.inf)

    for i in range(len(component_points)):
        for j_idx, j in enumerate(indices[i]):
            if i != j and j < len(component_points):  # 确保索引有效
                vec = component_points[j] - component_points[i]
                cos_angle = np.abs(np.dot(vec, main_axis)) / np.linalg.norm(vec)
                penalty = 5.0 if cos_angle < 0.3 else 1.0
                adj[i, j] = dists[i][j_idx] * penalty

    start_idx_in_component = np.where(component_indices == new_start_idx)[0]
    if len(start_idx_in_component) == 0:
        start_idx_in_component = [0]
        print("警告：使用分量内第一个点作为起点")

    geo_dists_component = dijkstra(adj, indices=[start_idx_in_component[0]], directed=False)[0]
    geo_dists_full = np.full(len(points), np.inf)
    geo_dists_full[component_indices] = geo_dists_component

    return geo_dists_full


def find_end_point(points, start_idx, labels, n_components):
    """根据连通区域找到终点"""
    start_component = labels[start_idx]

    if n_components == 1:
        geo_dists = geodesic_with_radial_constraint(points, start_idx)
        valid_mask = ~np.isinf(geo_dists)
        if np.any(valid_mask):
            end_idx = np.argmax(geo_dists[valid_mask])
            valid_indices = np.where(valid_mask)[0]
            return valid_indices[end_idx], geo_dists
        else:
            return start_idx, np.zeros(len(points))

    else:
        # 多连通分量：选择y均值最大的分量
        component_stats = []
        for comp_id in range(n_components):
            comp_mask = (labels == comp_id)
            comp_points = points[comp_mask]
            mean_y = np.mean(comp_points[:, 1])
            comp_size = np.sum(comp_mask)

            component_stats.append({
                'comp_id': comp_id,
                'mean_y': mean_y,
                'size': comp_size,
                'mask': comp_mask,
                'points': comp_points
            })

        component_stats.sort(key=lambda x: x['mean_y'], reverse=True)
        target_component = component_stats[0]
        geo_dists = geodesic_in_component(points, start_idx, target_component['mask'])
        comp_indices = np.where(target_component['mask'])[0]
        comp_geo_dists = geo_dists[comp_indices]

        valid_mask = ~np.isinf(comp_geo_dists)
        if np.any(valid_mask):
            max_idx_in_comp = np.argmax(comp_geo_dists[valid_mask])
            valid_indices = comp_indices[valid_mask]
            end_idx = valid_indices[max_idx_in_comp]
        else:
            center_point = np.mean(target_component['points'], axis=0)
            tree = KDTree(points)
            result = tree.query(center_point.reshape(1, -1), k=1)
            if isinstance(result, tuple) and len(result) == 2:
                dists, indices = result
                end_idx = indices[0][0]
            else:
                end_idx = result[1][0]

        return end_idx, geo_dists


def draw_line_3d(start, end, volume, value=3000):
    start = np.array(start, float)
    end = np.array(end, float)
    steps = int(np.max(np.abs(end - start))) + 1
    line = np.linspace(start, end, steps)
    for x, y, z in np.round(line).astype(int):
        if 0 <= x < volume.shape[0] and 0 <= y < volume.shape[1] and 0 <= z < volume.shape[2]:
            volume[x, y, z] = value
    return volume


def find_plane_intersection_line(plane1, plane2, ct_data, affine, header):
    a1, b1, c1, d1 = plane1
    a2, b2, c2, d2 = plane2
    direction = np.cross([a1, b1, c1], [a2, b2, c2])
    direction = direction / np.linalg.norm(direction)
    det = a1 * b2 - a2 * b1
    x = (b1 * d2 - b2 * d1) / det
    y = (a2 * d1 - a1 * d2) / det
    point = np.array([x, y, 0])
    p1 = point - direction * 3000
    p2 = point + direction * 3000
    vol = ct_data.copy()
    vol = draw_line_3d(p1, p2, vol)
    nii_vol = nib.Nifti1Image(vol, affine, header)
    return nii_vol

def save_slices_as_png(slices_data, slices_mask, slice_names, output_dir, patient_id):
    os.makedirs(output_dir, exist_ok=True)
    norm_data = ((slices_data - slices_data.min()) / (slices_data.max() - slices_data.min() + 1e-8) * 255).astype(
        np.uint8)
    rgb = np.zeros((*slices_data.shape, 3), dtype=np.uint8)
    for i in range(3):
        rgb[:, :, i] = norm_data
    # 应用mask颜色
    mask_condition = (slices_mask == 2) | (slices_mask == 3)
    rgb[mask_condition] = [0, 0, 255]  # 蓝色
    rgb[slices_mask == 0] = [0, 0, 0]
    rgb[(slices_mask == 0) & (slices_data == 3000)] = [0, 0, 0]
    img = Image.fromarray(rgb, 'RGB')
    output_path = f"{output_dir}/{patient_id}_{slice_names}.png"
    img.save(output_path)
    # print(f"输入: {slices_data.shape}, 输出PNG: {img.size}")
    return output_path


def nifti_to_pointcloud(
        # input_nifti_path,
        img, fdata, affine,
        output_dir=None,
        label_value=1,
        verbose=True
):
    """
    将NIfTI文件中指定label值的区域转换为TXT格式的点云（每行x y z）
    """
    label_coords = np.argwhere(fdata == label_value)
    homogeneous_coords = np.c_[label_coords, np.ones(len(label_coords))]
    world_coords = (affine @ homogeneous_coords.T).T[:, :3]
    return world_coords


def extract_mandible_inferior_points_from_txt(
        points,
        slice_thickness=1.0,
        separate_sides=True,
        num_nearest=5000,
):
    def extract_inferior_points(points, thickness, separate, nearest):
        """核心提取逻辑"""
        z_min, z_max = np.min(points[:, 2]), np.max(points[:, 2])
        slice_bins = np.arange(z_min, z_max + thickness, thickness)
        result = []

        for i in range(len(slice_bins) - 1):
            z_low, z_high = slice_bins[i], slice_bins[i + 1]
            slice_mask = (points[:, 2] >= z_low) & (points[:, 2] < z_high)
            slice_points = points[slice_mask]

            if len(slice_points) == 0:
                continue

            if separate:
                x_mid = np.median(points[:, 0])
                for side_points in [slice_points[slice_points[:, 0] < x_mid],
                                    slice_points[slice_points[:, 0] >= x_mid]]:
                    if len(side_points) > 0:
                        ref_idx = np.argmin(side_points[:, 1])
                        dists = np.linalg.norm(side_points - side_points[ref_idx], axis=1)
                        nearest_idx = np.argpartition(dists, min(nearest, len(dists) - 1))[:nearest]
                        result.extend(side_points[nearest_idx])
            else:
                ref_idx = np.argmin(slice_points[:, 1])
                dists = np.linalg.norm(slice_points - slice_points[ref_idx], axis=1)
                nearest_idx = np.argpartition(dists, min(nearest, len(dists) - 1))[:nearest]
                result.extend(slice_points[nearest_idx])

        return np.array(result)

    if len(points) == 0:
        raise ValueError("点云数据为空")
    # 提取下缘点
    inferior_points = extract_inferior_points(
        points,
        thickness=slice_thickness,
        separate=separate_sides,
        nearest=num_nearest
    )

    return inferior_points


def process_point_cloud_txt(
        points
):
    """
    处理TXT格式点云文件，提取X最大和Z最小点之间的区域
    """

    def crop_between_points(points, p1, p2):
        """裁剪两个点之间的区域"""
        z_min = min(p1[2], p2[2])
        z_max = max(p1[2], p2[2])
        mask = (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
        return points[mask]

    # 找到X最大和Z最小的点
    x_max_index = np.argmax(points[:, 1])
    z_max_index = np.argmax(points[:, 2])
    z_min_index = np.argmin(points[:, 2])
    z_mid_index = (int)((z_min_index + z_max_index) / 2)
    x_max_index = z_mid_index
    x_max_point = points[x_max_index]
    z_min_point = points[z_min_index]
    boundary_points = np.array([x_max_point, z_min_point])
    cropped_points = crop_between_points(points, x_max_point, z_min_point)
    return cropped_points


def extract_and_interpolate_centerline(
        points,
        # output_file=None,
        pca_components=2,
        num_steps=80,
        smooth_window=5,
        interpolation_step=0.1,
):
    """
    从点云提取中心线并插值（输入输出均为TXT）
    """
    try:
        # PCA找主方向
        pca = PCA(n_components=pca_components)
        points_2d = points[:, :2]  # 假设主要在XY平面
        pca.fit(points_2d)
        main_dir = pca.components_[0]  # 第一主成分方向

        projections = np.dot(points_2d, main_dir)
        proj_min, proj_max = np.min(projections), np.max(projections)
        step_size = (proj_max - proj_min) / num_steps

        centerline = []
        for i in range(num_steps):
            current_pos = proj_min + i * step_size
            window_center = current_pos + step_size / 2
            mask = (projections >= window_center - step_size) & \
                   (projections <= window_center + step_size)
            window_points = points[mask]

            if len(window_points) > 0:
                mean_pt = np.mean(window_points, axis=0)
                median_pt = np.median(window_points, axis=0)
                center_pt = 0.7 * mean_pt + 0.3 * median_pt
                centerline.append(center_pt)
        centerline = np.array(centerline)

        # 平滑处理
        if len(centerline) > smooth_window:
            centerline[:, 0] = savgol_filter(centerline[:, 0], smooth_window, 2)
            centerline[:, 1] = savgol_filter(centerline[:, 1], smooth_window, 2)
            centerline[:, 2] = savgol_filter(centerline[:, 2], smooth_window, 2)

        # ================== 中心线插值 ==================
        def interpolate_centerline(points, step_size):
            """精确插值函数"""
            diff = np.diff(points, axis=0)
            dist = np.linalg.norm(diff, axis=1)
            cum_dist = np.insert(np.cumsum(dist), 0, 0)

            fx = interp1d(cum_dist, points[:, 0], kind='linear')
            fy = interp1d(cum_dist, points[:, 1], kind='linear')
            fz = interp1d(cum_dist, points[:, 2], kind='linear')
            new_dist = np.arange(0, cum_dist[-1], step_size)
            return np.column_stack([fx(new_dist), fy(new_dist), fz(new_dist)])

        interpolated = interpolate_centerline(centerline, interpolation_step)
        return interpolated

    except Exception as e:
        raise RuntimeError(f"中心线提取失败: {str(e)}")


def spline_fit_3d(points, smoothing=0.1, degree=3):
    """
    三维样条曲线拟合
    """
    points_t = points.T
    tck, u = splprep(points_t, s=smoothing * len(points), k=degree)
    # 生成更密集的采样点
    u_new = np.linspace(0, 1, 200)  # 增加采样点数量
    curve_points = splev(u_new, tck)
    curve_points = np.array(curve_points).T
    # 计算导数（切线方向）
    derivative_points = splev(u_new, tck, der=1)
    tangents = np.array(derivative_points).T

    return curve_points, tangents, tck


def create_point_cloud(points, color):
    """创建Open3D点云对象"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color(color)
    return pcd


def create_line_set(points, color):
    """创建Open3D线集对象"""
    if len(points) < 2:
        return None

    # 创建连线
    lines = [[i, i + 1] for i in range(len(points) - 1)]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.paint_uniform_color(color)

    return line_set


def rotation_matrix_from_vectors(vec1, vec2):
    """计算两个向量之间的旋转矩阵"""
    a, b = vec1 / np.linalg.norm(vec1), vec2 / np.linalg.norm(vec2)
    v = np.cross(a, b)
    c = np.dot(a, b)

    if np.allclose(v, 0):  # 向量平行
        return np.eye(3) if c > 0 else -np.eye(3)  # 反向

    vx = np.array([[0, -v[2], v[1]],
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]])

    R = np.eye(3) + vx + vx @ vx * (1 / (1 + c))
    return R


def calculate_fitting_metrics(original_points, fitted_curve):
    """计算拟合质量指标"""
    # 创建拟合曲线的KD树
    tree = KDTree(fitted_curve)

    # 计算每个原始点到拟合曲线的距离
    distances, _ = tree.query(original_points)

    metrics = {
        'max_error': np.max(distances),
        'mean_error': np.mean(distances),
        'std_error': np.std(distances),
        'rmse': np.sqrt(np.mean(distances ** 2))
    }

    return metrics


def pointcloud_to_nifti(
        points,
        output_nifti_path=None,
        original_shape=(512, 512, 404),
        voxel_size=0.3,  # 0.3mm 间距
        label_value=1,
        verbose=True
):
    """
    将TXT格式的点云转换回NIfTI文件，使用固定的图像尺寸和空间信息
    """
    nifti_data = np.zeros(original_shape)
    affine = np.eye(4)
    affine[0, 0] = -voxel_size  # X轴（R→L）：-0.3
    affine[1, 1] = -voxel_size  # Y轴（A→P）：-0.3
    affine[2, 2] = voxel_size  # Z轴（I→S）：0.3
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    center_coords = (min_coords + max_coords) / 2

    image_center = np.array([
        -original_shape[0] * voxel_size / 2,  # X方向（负）
        -original_shape[1] * voxel_size / 2,  # Y方向（负）
        original_shape[2] * voxel_size / 2  # Z方向（正）
    ])

    affine[:3, 3] = center_coords - image_center
    inv_affine = np.linalg.inv(affine)
    homogeneous_points = np.c_[points, np.ones(len(points))]
    voxel_coords = (inv_affine @ homogeneous_points.T).T[:, :3]
    voxel_coords = np.round(voxel_coords).astype(int)
    valid_mask = (
            (voxel_coords[:, 0] >= 0) & (voxel_coords[:, 0] < original_shape[0]) &
            (voxel_coords[:, 1] >= 0) & (voxel_coords[:, 1] < original_shape[1]) &
            (voxel_coords[:, 2] >= 0) & (voxel_coords[:, 2] < original_shape[2])
    )

    valid_coords = voxel_coords[valid_mask]

    nifti_data[valid_coords[:, 0], valid_coords[:, 1], valid_coords[:, 2]] = label_value

    if len(valid_coords) > 0:
        nifti_data = ndimage.binary_dilation(nifti_data, structure=np.ones((3, 3, 3))).astype(nifti_data.dtype)
    nifti_img = nib.Nifti1Image(nifti_data, affine)
    return nifti_img


# 保存牙弓标签函数
def save_dental_arch_label(label_data, label_img, z_slice, output_path):
    """
    """
    new_label = label_data.copy().astype(np.uint8)
    arch_mask = (label_data == 6)

    if np.any(arch_mask):
        z_coords = np.where(arch_mask)[2]
        if len(z_coords) > 0:
            current_center_z = np.median(z_coords).astype(int)
            z_offset = z_slice - current_center_z
            coords = np.argwhere(arch_mask)
            for (x, y, z) in coords:
                new_z = z + z_offset
                if 0 <= new_z < new_label.shape[2]:
                    new_label[x, y, z] = 1
                    new_label[x, y, new_z] = 6
        new_img = nib.Nifti1Image(new_label, label_img.affine, label_img.header)
        new_img.header['cal_min'] = 0
        new_img.header['cal_max'] = 6

        # 保存文件
        nib.save(new_img, output_path)
    else:
        print("⚠️ 未找到label=6的牙弓结构")


def keep_largest_connected_component(mask):
    """
    保留掩码中最大的连通组件，去除其他部分
    """
    labeled_array, num_features = label(mask)
    if num_features == 0:
        return mask
    component_sizes = np.bincount(labeled_array.ravel())
    component_sizes[0] = 0
    largest_component = np.argmax(component_sizes)
    largest_mask = (labeled_array == largest_component)
    return largest_mask


# 定义关键点查找函数
def find_key_points(label_data, case_id):
    left_canal = np.argwhere(label_data == 2)
    right_canal = np.argwhere(label_data == 3)

    # 处理左侧下颌管
    if len(left_canal) > 0:
        left_mask = np.zeros_like(label_data, dtype=bool)
        left_mask[tuple(left_canal.T)] = True  # 将坐标转换为掩码
        left_mask = keep_largest_connected_component(left_mask)
        left_canal = np.argwhere(left_mask)  # 更新为处理后的坐标

    # 处理右侧下颌管
    if len(right_canal) > 0:
        right_mask = np.zeros_like(label_data, dtype=bool)
        right_mask[tuple(right_canal.T)] = True
        right_mask = keep_largest_connected_component(right_mask)
        right_canal = np.argwhere(right_mask)

    # 查找下颌骨区域
    mandible = np.argwhere(label_data == 1)
    if len(mandible) > 0:
        # x坐标最大（即第一列最大）
        max_x_idx = np.argmin(mandible[:, 1])
        mandible_max_x_point = mandible[max_x_idx]
    else:
        mandible_max_x_point = None

    if len(left_canal) == 0 or len(right_canal) == 0:
        raise ValueError(f"案例{case_id}: 无法找到下颌管，请检查标签数据中是否存在标签2和3")

    # 定位颏孔
    def find_lowest_point(coords):
        max_z = np.max(coords[:, 1])  # 找到所有点中，z坐标最大的值
        candidates = coords[coords[:, 1] == max_z]  # 找到z坐标等于最大值的所有点
        return np.mean(candidates, axis=0)  # 计算这些点的平均位置，得到最前面点的中心

    # 定位下颌孔 find_lowest_point
    def find_frontmost_point(coords):
        min_z = np.min(coords[:, 1])
        candidates = coords[coords[:, 1] == min_z]
        return np.mean(candidates, axis=0)

    def find_lowest_point_z(coords):
        min_z = np.min(coords[:, 2])
        candidates = coords[coords[:, 2] == min_z]
        return np.mean(candidates, axis=0)

    left_lowest = find_lowest_point(left_canal)
    right_lowest = find_lowest_point(right_canal)

    left_lowest_z = find_lowest_point_z(left_canal)
    right_lowest_z = find_lowest_point_z(right_canal)
    plane_z = (left_lowest_z[2] + right_lowest_z[2]) / 2 - 15

    left_mental = find_frontmost_point(left_canal)
    right_mental = find_frontmost_point(right_canal)

    left_midpoint = (left_mental + left_lowest) / 2
    right_midpoint = (right_mental + right_lowest) / 2

    plan_z1 = (left_mental[2] + right_mental[2]) / 2 + 10

    def calculate_plane_normal(points):
        """计算三点确定的平面法向量"""
        v1 = points[1] - points[0]
        v2 = points[2] - points[0]
        normal = np.cross(v1, v2)
        return normal / np.linalg.norm(normal)  # 单位化

    def get_mandible_points_in_plane(label_data, plane_points, plane_normal, tolerance=3.0):
        """获取平面内的下颌骨点（标签为1的点）"""
        mandible_points = np.argwhere(label_data == 1)  # 假设下颌骨标签为1

        if len(mandible_points) == 0:
            return np.array([])

        # 计算点到平面的距离
        distances = np.abs(np.dot(mandible_points - plane_points[0], plane_normal))

        # 选择距离平面在一定范围内的点
        in_plane_mask = distances <= tolerance
        return mandible_points[in_plane_mask]

    return {
        'left_mental': left_mental,
        'right_mental': right_mental,
        'left_midpoint': left_midpoint,
        'right_midpoint': right_midpoint,
        'left_canal': left_lowest,
        'right_canal': right_lowest,
        'plane_z': plane_z,
        'plane_z1': plan_z1,
        # "mandible_points":mandible_points
    }


def merge_xiayuanxiang(nii_img, label_data, output_dir, label_pattern="*.nii.gz", label_affine=None, header=None,
                       filename=None):
    """
    将下颌数据合并到标签数据中
    """
    os.makedirs(output_dir, exist_ok=True)
    results = []
    label_data = label_data.astype(np.uint8)

    xiayuanxiang_img = nii_img
    xiayuanxiang_data = xiayuanxiang_img.get_fdata()
    xiayuanxiang_data = xiayuanxiang_data.astype(np.uint8)
    xiayuanxiang_affine = xiayuanxiang_img.affine
    if not np.allclose(label_affine, xiayuanxiang_affine, atol=1e-3):
        # print("⚠️ affine矩阵不一致，将调整下颌数据空间信息")

        transform = np.linalg.inv(xiayuanxiang_affine) @ label_affine

        xiayuanxiang_data = affine_transform(
            xiayuanxiang_data,
            matrix=transform[:3, :3],
            offset=transform[:3, 3],
            output_shape=label_data.shape,
            order=0  # 最近邻插值
        )
        # print("✅ 已完成空间调整")
    else:
        print("✅ affine矩阵一致，无需调整")

    xiayuanxiang_mask = (xiayuanxiang_data > 0)

    new_data = label_data.copy()  # 创建副本
    new_data[xiayuanxiang_mask] = 6  # 只修改mask区域

    new_img = nib.Nifti1Image(new_data, label_affine, header)

    new_img.header['cal_min'] = np.min(new_data)
    new_img.header['cal_max'] = np.max(new_data)
    new_img.header['glmin'] = np.min(new_data)
    new_img.header['glmax'] = np.max(new_data)
    filename = filename + '.nii.gz'
    output_path = os.path.join(output_dir, filename)
    # nib.save(new_img, output_path)
    # print(filename, output_path, new_data.shape)
    return new_img


def find_end_point(points, start_idx, labels, n_components):
    """根据连通区域找到终点"""
    start_component = labels[start_idx]

    if n_components == 1:
        geo_dists = geodesic_with_radial_constraint(points, start_idx)
        valid_mask = ~np.isinf(geo_dists)
        if np.any(valid_mask):
            end_idx = np.argmax(geo_dists[valid_mask])
            valid_indices = np.where(valid_mask)[0]
            return valid_indices[end_idx], geo_dists
        else:
            return start_idx, np.zeros(len(points))

    else:
        # 多连通分量：选择y均值最大的分量
        component_stats = []
        for comp_id in range(n_components):
            comp_mask = (labels == comp_id)
            comp_points = points[comp_mask]
            mean_y = np.mean(comp_points[:, 1])
            comp_size = np.sum(comp_mask)

            component_stats.append({
                'comp_id': comp_id,
                'mean_y': mean_y,
                'size': comp_size,
                'mask': comp_mask,
                'points': comp_points
            })

        component_stats.sort(key=lambda x: x['mean_y'], reverse=True)
        target_component = component_stats[0]
        # print(f"选择连通分量 {target_component['comp_id']}, y均值: {target_component['mean_y']:.2f}, 点数: {target_component['size']}")
        geo_dists = geodesic_in_component(points, start_idx, target_component['mask'])
        comp_indices = np.where(target_component['mask'])[0]
        comp_geo_dists = geo_dists[comp_indices]

        valid_mask = ~np.isinf(comp_geo_dists)
        if np.any(valid_mask):
            max_idx_in_comp = np.argmax(comp_geo_dists[valid_mask])
            valid_indices = comp_indices[valid_mask]
            end_idx = valid_indices[max_idx_in_comp]
        else:
            center_point = np.mean(target_component['points'], axis=0)
            tree = KDTree(points)
            result = tree.query(center_point.reshape(1, -1), k=1)
            if isinstance(result, tuple) and len(result) == 2:
                dists, indices = result
                end_idx = indices[0][0]
            else:
                end_idx = result[1][0]

        return end_idx, geo_dists


def find_connected_components(points, k=15):
    """找到点云的连通分量"""
    tree = KDTree(points)
    dists, indices = tree.query(points, k=k + 1)
    adj = np.full((len(points), len(points)), np.inf)
    for i in range(len(points)):
        for j_idx, j in enumerate(indices[i]):
            if i != j:
                adj[i, j] = dists[i][j_idx]
    n_components, labels = connected_components(adj, directed=False)
    return labels, n_components


def geodesic_in_component(points, start_idx, component_mask, k=15):
    """在指定连通分量内计算测地距离"""
    component_points = points[component_mask]
    component_indices = np.where(component_mask)[0]

    print(f"分量内点数: {len(component_points)}, 起点是否在分量内: {start_idx in component_indices}")
    if start_idx not in component_indices:
        tree = KDTree(component_points)
        start_point = points[start_idx].reshape(1, -1)
        result = tree.query(start_point, k=1)
        print(f"KDTree查询结果类型: {type(result)}, 长度: {len(result)}")
        if isinstance(result, tuple) and len(result) == 2:
            dists, indices = result
            nearest_idx = indices[0]  # 第一个元素的索引
        else:
            dists = result[0]
            indices = result[1]
            nearest_idx = indices[0]
        new_start_idx = component_indices[nearest_idx]
    else:
        # 找到起点在分量内的位置
        new_start_idx_in_comp = np.where(component_indices == start_idx)[0][0]
        new_start_idx = start_idx
    centered = component_points - component_points.mean(axis=0)
    covariance = centered.T @ centered
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    main_axis = eigenvectors[:, 2]  # 主方向

    tree_component = KDTree(component_points)
    k_val = min(k + 1, len(component_points))
    dists, indices = tree_component.query(component_points, k=k_val)

    adj = np.full((len(component_points), len(component_points)), np.inf)

    for i in range(len(component_points)):
        for j_idx, j in enumerate(indices[i]):
            if i != j and j < len(component_points):  # 确保索引有效
                vec = component_points[j] - component_points[i]
                cos_angle = np.abs(np.dot(vec, main_axis)) / np.linalg.norm(vec)
                penalty = 5.0 if cos_angle < 0.3 else 1.0
                adj[i, j] = dists[i][j_idx] * penalty

    start_idx_in_component = np.where(component_indices == new_start_idx)[0]
    if len(start_idx_in_component) == 0:
        start_idx_in_component = [0]
        print("警告：使用分量内第一个点作为起点")

    geo_dists_component = dijkstra(adj, indices=[start_idx_in_component[0]], directed=False)[0]
    geo_dists_full = np.full(len(points), np.inf)
    geo_dists_full[component_indices] = geo_dists_component

    return geo_dists_full


def find_max_y_point(points):
    """
    找到点云中Y坐标最大的点
    """
    return points[np.argmax(points[:, 1])]


def geodesic_with_radial_constraint(points, start_idx, k=15):
    """计算带径向约束的测地距离"""
    centered = points - points.mean(axis=0)
    main_axis = np.linalg.eigh(centered.T @ centered)[1][:, 2]  # 主方向

    tree = KDTree(points)
    dists, indices = tree.query(points, k=k + 1)

    adj = np.full((len(points), len(points)), np.inf)

    for i in range(len(points)):
        for j_idx, j in enumerate(indices[i]):
            if i != j:
                vec = points[j] - points[i]
                cos_angle = np.abs(np.dot(vec, main_axis)) / np.linalg.norm(vec)
                penalty = 5.0 if cos_angle < 0.3 else 1.0
                adj[i, j] = dists[i][j_idx] * penalty

    return dijkstra(adj, indices=[start_idx], directed=False)[0]


def find_min_dist_poin(point, yagong_point):
    point = np.asarray(point)
    yagong_point = np.asarray(yagong_point)
    distances = np.linalg.norm(yagong_point - point, axis=1)
    min_index = np.argmin(distances)
    min_distance = distances[min_index]
    min_point = yagong_point[min_index]
    return min_point

def get_houya_col(pm_len):
    key_lens = np.array([0,6,8])

    #红色 黄色 绿色
    key_colors = np.array([
        [255, 0, 0],
        [255, 178, 102],
        # [255,255,0],
        [0,255,0]
    ])
    ids = np.searchsorted(key_lens, pm_len) - 1
    idx = min(max(ids,0), len(key_lens)-2)
    t = (pm_len - key_lens[idx]) / (key_lens[idx+1] - key_lens[idx])
    col = (1-t)*key_colors[idx] + t*key_colors[idx+1]
    print('pm_len:', pm_len, 'col',col)
    return (col/255.0).tolist()


def paint_between_planes(points, plane1, plane2, color1, color2):
    max_dist = 2
    a1, b1, c1, d1 = plane1
    a2, b2, c2, d2 = plane2
    n1 = np.array([a1, b1, c1])
    n2 = np.array([a2, b2, c2])
    s1 = points @ n1 + d1
    s2 = points @ n2 + d2
    between_mask = (s1 * s2) <= 0
    between_mask &= (np.abs(s1) <= max_dist) & (np.abs(s2) <= max_dist)
    idx = np.where(between_mask)[0]
    if len(idx) == 0:
        return between_mask, None
    d1_abs = np.abs(s1[idx])
    d2_abs = np.abs(s2[idx])
    w = d1_abs / (d1_abs + d2_abs + 1e-8)
    blended = (
        (1 - w)[:, None] * color1 +
        w[:, None] * color2
    )
    return between_mask, blended


def get_star_end(lmax_midpoint, yagong_filt_xl, lkq_midpoint):
    lsta = np.argwhere(yagong_filt_xl == lkq_midpoint)[0][0]
    lend = np.argwhere(yagong_filt_xl == lmax_midpoint)[0][0]
    inde_sta = lsta # 找起点
    dist = 0
    for i in range(lsta, -1, -1):
        dist_sta = np.linalg.norm(yagong_filt_xl[lsta] - yagong_filt_xl[i])
        dist += dist_sta
        if dist_sta > 2:
            inde_sta = i
            dist = 0
            break
    inde_end = lend # 找终点
    for i in range(lend, len(yagong_filt_xl)):
        dist_end = np.linalg.norm(yagong_filt_xl[lend] - yagong_filt_xl[i])
        if dist_end > 2:
            inde_end = i
            dist = 0
            break
    return inde_sta, inde_end

def get_clo_list(bone_len, min_val, max_val):
    if bone_len <= 6:
        # 橙 -> 红（局部 0~1）
        t = (bone_len - min_val) / (6 - min_val)
        t = np.clip(t, 0, 1)
        r = 1.0
        g = 1 - t
        b = 0.4
    elif 6 < bone_len <= 8:
        # 黄 -> 绿
        t = (bone_len - 6) / (8 - 6)
        t = np.clip(t, 0, 1)
        r = 1 - t
        g = 1.0
        b = 0.4
    else:
        r = 0.7
        g = 1.0
        b = 0.2
    return [r, g, b]


