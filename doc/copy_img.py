import os
import shutil

def copy_img(path1, path2):
    for root, dirs, files in os.walk(path1):
        for file in files:  
            if file.endswith('.png'):  
                # print(root, dirs,file)  
                shutil.copy(os.path.join(root,file),os.path.join(path2,file))

copy_img(r'C:\yuechen\code\wuyahe\1.code\2.data-缩放\screenshot\pca-sum\qianya',
         r'C:\yuechen\code\wuyahe\1.code\2.data-缩放\screenshot\pca-sum\qianya\img')