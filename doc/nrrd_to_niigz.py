import argparse
from pathlib import Path

import SimpleITK as sitk
import os

def convert_one_file(src_path, dst_path): 
    """Convert one .nrrd file to .nii.gz."""
    image = sitk.ReadImage(src_path)
    
    sitk.WriteImage(image, dst_path)


path = r"F:\wuyahe\data\1-30\1-30"
path2 = r"F:\wuyahe\data\1-30\niigz"

for file in os.listdir(path):
    
    if file.endswith(".nrrd"):
        # index = file.split(".")[0]
        # src_path = os.path.join(path,file)
        # dst_path = os.path.join(path2,index+".nii.gz")
        # convert_one_file(src_path,dst_path)
        index = int(file.split(".")[0])
        index_str = f"{index:03d}"

        src_path = os.path.join(path, file)
        dst_path = os.path.join(path2, index_str + ".nii.gz")
        convert_one_file(src_path,dst_path)