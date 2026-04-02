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
        prefix = file.split('_')[1]
        os.rename(os.path.join(path, file), os.path.join(path,prefix))
        print(file,prefix)


rename_files(r'C:\Users\yuechen\Desktop\fsdownload\pred')