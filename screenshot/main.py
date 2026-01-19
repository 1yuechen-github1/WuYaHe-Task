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

# 设置中文字体和输出路径
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False  

def process_single_ct(ct_path, base_dir, output_base_dir):
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
    output_dir = os.path.join(output_base_dir, "screenshot1", filename)
    os.makedirs(output_dir, exist_ok=True)
    output_files_dir = os.path.join(base_dir, "output", filename)
    os.makedirs(output_files_dir, exist_ok=True)

    ct_img = nib.load(ct_path)
    ct_data = ct_img.get_fdata()
    label_img = nib.load(label_path)
    label_data = label_img.get_fdata()

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
        ct_data = rescale(ct_data, scale, order=3,preserve_range=True,mode='reflect',anti_aliasing=False)
        fdata = rescale(fdata, scale, order=0,preserve_range=True,mode='edge',anti_aliasing=False )
        n_affine = affine.copy()
        for i in range(3):
            n_affine[:3, i] *= spacing / img_spacing[i]
        affine = n_affine

    xhg_points = nifti_to_pointcloud(img,fdata,affine,output_dir = output_dir,label_value =1) 
    left_xiaheguan_data = nifti_to_pointcloud(img,fdata,affine,output_dir = output_dir,label_value =2) 
    right_xiaheguan_data = nifti_to_pointcloud(img,fdata,affine,output_dir = output_dir,label_value =3) 

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
    left_z_centroid = compute_y_mid_point(left_xiaheguan_data)
    right_z_centroid = compute_y_mid_point(right_xiaheguan_data)
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
        # 4. 保存拟合结果（可选）/ 下缘线
        # np.savetxt(os.path.join(output_dir, f"{filename}_step4_filt.txt"), fitted_curve, fmt='%.6f')    
    
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
    print(offset_z , optimal_offset, abs(point[2] - (left_kekong[2] + right_kekong[2]) / 2))
    yagong_point = xiayuanxiang_path_data.copy()
    yagong_point[:, 2] = original_z + optimal_offset * 1
        
    nii_img = pointcloud_to_nifti(
        points=yagong_point,
        original_shape=shape,
        voxel_size=spacing,
        label_value=1,
        verbose=True,
    )
    save_path = os.path.join(output_dir, filename+'.nii.gz')
    new_img = merge_xiayuanxiang(
    nii_img=nii_img,
    label_data=fdata,
    output_dir=output_dir,
    label_affine=affine,
    header=header,
    filename = filename
    )        
    save_path = os.path.join(output_dir, filename+'.nii.gz')
    print(filename,save_path,shape,abs(point[2] - (left_kekong[2] + right_kekong[2]) / 2))
    nib.save(new_img, save_path)

 
    left_xiayuanxiang_path_data, right_xiayuanxiang_path_data = split_along_x(xiayuanxiang_path_data)      
    left_plane_coeffs, left_closest_point = compute_shortest_distance_to_curve_with_perpendicular_plane(
         left_xiayuanxiang_path_data, left_z_centroid
    )
    right_plane_coeffs, right_closest_point = compute_shortest_distance_to_curve_with_perpendicular_plane(
        right_xiayuanxiang_path_data, right_z_centroid
    )
    left_plane_xiahekong_coeffs, left_xiahekong_closest_point = compute_shortest_distance_to_curve_with_perpendicular_plane(
        left_xiayuanxiang_path_data, left_kekong
    )
    right_plane_xiahekong_coeffs, right_closest_xiahekong_point = compute_shortest_distance_to_curve_with_perpendicular_plane(
        right_xiayuanxiang_path_data, right_kekong
    )
    max_y = find_max_y_point(yagong_point)
    three_points_plane_coeffs = create_plane_from_three_points(
        left_kekong, right_kekong, max_y
    )
    plane_coeffs_mid, closest_mid_point = get_plane(
        curve_points=yagong_point, mid_point=max_y
    )

    # 可视化
    # visualize_with_open3d(left_xiaheguan_data, right_xiaheguan_data,
    #                          yagong_point, yagong_point,
    #                          left_plane_coeffs, right_plane_coeffs,
    #                          left_closest_point, right_closest_point,
    #                          left_plane_xiahekong_coeffs,right_plane_xiahekong_coeffs,
    #                          left_xiahekong_closest_point,right_closest_xiahekong_point,
    #                          three_points_plane_coeffs,max_y)

    # visualize_with_open3d(left_xiaheguan_data, right_xiaheguan_data,
    #                          yagong_point, yagong_point,

    #                          left_plane_coeffs, right_plane_coeffs,
    #                          left_kekong, right_kekong,

    #                          left_plane_xiahekong_coeffs,right_plane_xiahekong_coeffs,
    #                          stl_point,str_point,

    #                          three_points_plane_coeffs,max_y)

    
    # 坐标转化为nifti
    left_plane_coeffs_nifti, left_center_nifti = trans_to_nifti(
        left_plane_coeffs, left_closest_point, ct_img, affine)
    right_plane_coeffs_nifti, right_center_nifti = trans_to_nifti(
        right_plane_coeffs, right_closest_point, ct_img,affine)
    
    left_xiahekong_coeffs_nifti, left_xiahekong_center_nifti = trans_to_nifti(
        left_plane_xiahekong_coeffs, left_xiahekong_closest_point, ct_img,affine)
    right_xiahekong_coeffs_nifti, right_xiahekong_center_nifti = trans_to_nifti(
        right_plane_xiahekong_coeffs, right_closest_xiahekong_point, ct_img,affine)
    
    mid_point_coeffs_nifti, mid_center_nifti = trans_to_nifti(
        plane_coeffs_mid, closest_mid_point, ct_img,affine)
    
    # 新平面：左右颏孔和max_x_point构成的平面
    three_points_plane_nifti, three_points_center_nifti = trans_to_nifti(
        three_points_plane_coeffs, max_y, ct_img,affine)
    high_hu_img = find_plane_intersection_line(
        mid_point_coeffs_nifti,      
        three_points_plane_nifti,    
        ct_data,affine,header
    )   


    # 提取截面
    # left_mid_slice = extract_slice(
    #     ct_data, fdata, left_plane_coeffs_nifti, left_center_nifti)
    # right_mid_slice = extract_slice(
    #     ct_data, fdata, right_plane_coeffs_nifti, right_center_nifti)
    # left_xiahekong_slice = extract_slice(
    #     ct_data, fdata, left_xiahekong_coeffs_nifti, left_xiahekong_center_nifti)
    # right_xiahekong_slice = extract_slice(
    #     ct_data, fdata, right_xiahekong_coeffs_nifti, right_xiahekong_center_nifti)
    # mid_slice = extract_slice(
    #     high_hu_img.get_fdata(), fdata, mid_point_coeffs_nifti, mid_center_nifti)
    
    
    # # 旋转180度
    # left_xiahekong_slice = np.rot90(left_xiahekong_slice, 2)
    # left_mid_slice = np.rot90(left_mid_slice, 2)
    # slices_data = [ left_mid_slice, right_mid_slice, left_xiahekong_slice, right_xiahekong_slice,mid_slice]
    # slice_names = [ '左侧中间截面', '右侧中间截面', '左侧颏孔截面', '右侧颏孔截面','中间截面']        
    # save_slices_as_png(slices_data, slice_names, output_dir, patient_id=filename)    
    # print(f" {filename} 处理完成")

    left_mid_slice,left_mid_mk = extract_slice(
        ct_data, fdata, left_plane_coeffs_nifti, left_center_nifti)
    right_mid_slice,right_mid_mk = extract_slice(
        ct_data, fdata, right_plane_coeffs_nifti, right_center_nifti)
    left_xiahekong_slice,left_xiahekong_mk = extract_slice(
        ct_data, fdata, left_xiahekong_coeffs_nifti, left_xiahekong_center_nifti)
    right_xiahekong_slice,right_xiahekong_mk = extract_slice(
        ct_data, fdata, right_xiahekong_coeffs_nifti, right_xiahekong_center_nifti)
    # mid_slice, mid_mk = extract_slice(
    #     high_hu_img.get_fdata(), fdata, mid_point_coeffs_nifti, mid_center_nifti)
    mid_slice, mid_mk = extract_slice(
        ct_data, fdata, mid_point_coeffs_nifti, mid_center_nifti)
    
    
    # 旋转180度
    # left_xiahekong_slice = np.rot90(left_xiahekong_slice, 2)
    # left_mid_slice = np.rot90(left_mid_slice, 2)
    # left_xiahekong_mk = np.rot90(left_xiahekong_mk, 2)
    # left_mid_mk = np.rot90(left_mid_mk, 2)
    slices_data = [ left_mid_slice, right_mid_slice, left_xiahekong_slice, right_xiahekong_slice,mid_slice]
    slices_mask = [left_mid_mk,right_mid_mk,left_xiahekong_mk,right_xiahekong_mk,mid_mk]
    slice_names = [ '左侧中间截面', '右侧中间截面', '左侧颏孔截面', '右侧颏孔截面','中间截面']        
    save_slices_as_png(slices_data,slices_mask ,slice_names, output_dir, patient_id=filename)    
    print(f" {filename} 处理完成")    

if __name__ == "__main__":
    base_dir = "data"
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
        success = process_single_ct(ct_path, base_dir, output_base_dir)
