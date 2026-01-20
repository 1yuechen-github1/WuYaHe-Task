import os
import shutil

def copy_img(path1):
    for file in os.listdir(path1):
        prex = file.split(".")[0]
        prex = prex.split("_")[0]
        print(prex,file)

copy_img(r'C:\yuechen\code\wuyahe\1.code\2.data-缩放\screenshot\qianya-pca\pca')