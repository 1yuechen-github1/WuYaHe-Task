# import os

# def rename_files(path):
#     for file in os.listdir(path):
#         if file.endswith('左侧中间截面.png'):
#             print(file.split('_')[0],file)
#             os.rename(os.path.join(path, file), os.path.join(path, f'{file.split("_")[0]}_left_middle.png'))
#         elif file.endswith('右侧中间截面.png'):
#             print(file.split('_')[0],file)
#             os.rename(os.path.join(path, file), os.path.join(path, f'{file.split("_")[0]}_right_middle.png'))

# rename_files(r'data\mid\origin\test\3')



import os



def rename_files(path):
    for file in os.listdir(path):
        prefix = file[0:3]
        # os.rename(os.path.join(path, file), os.path.join(path,f"WuYaHe_{prefix}.nii.gz"))
        if file.endswith("_0000.nii.gz"):
            continue
        else:
            os.rename(os.path.join(path, file), os.path.join(path,f"WuYaHe_{prefix}_0000.nii.gz"))

rename_files(r'F:\wuyahe\data\imagesTr')