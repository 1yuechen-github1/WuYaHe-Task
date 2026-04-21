import os
import shutil

# def copy_img(path1, path2):
#     for root, dirs, files in os.walk(path1):
#         for file in files:  
#             if file.endswith('.png'):  
#                 # print(root, dirs,file)  
#                 shutil.copy(os.path.join(root,file),os.path.join(path2,file))

# copy_img(r'C:\yuechen\code\wuyahe\1.code\2.data-缩放\screenshot\origin\houya',
#          r'C:\yuechen\code\wuyahe\1.code\2.data-缩放\screenshot\origin\img')


def copy_img(path1,path2):
    for file in os.listdir(path1):
        path3 = os.path.join(path1,file)
        for file1 in os.listdir(path3):
            # if file1.endswith('.png'):
            print(file1,"done!")
            path4 = os.path.join(path3,file1)
            shutil.copy(path4,os.path.join(path2,file,file1))


copy_img(r'C:\yuechen\code\wuyahe\1.code\2.data-缩放\screenshot1\kekong\left',
         r'C:\yuechen\code\wuyahe\1.code\2.data-缩放\screenshot1\kekong\right')