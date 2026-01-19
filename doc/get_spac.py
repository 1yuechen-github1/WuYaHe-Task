
import os
import shutil
import nibabel as nib

def get_spac(path1):
    for file in os.listdir(path1):
        if file.endswith(".nii.gz"):
            data = nib.load(os.path.join(path1, file))
            print(file,data.header.get_zooms())


get_spac(r"C:\yuechen\code\wuyahe\1.code\1.data\data\ct")