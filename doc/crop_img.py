import os
from PIL import Image

def crop_img(path):
    path2 = r"C:\yuechen\code\wuyahe\2.data\260312颏孔分类\260312颏孔分类\crop\3"
    for file in os.listdir(path):
        if file.endswith('.png'):
            print(file)
            img_path = os.path.join(path, file)
            img = Image.open(img_path)
            # 150 * 220
            cropped_img = img.crop((140, 160, 290, 380))  # (left, upper, right, lower) 
            save_path = os.path.join(path2, file)
            cropped_img.save(save_path)
            print(f'Cropped image saved to {save_path}')
    

crop_img(r"C:\yuechen\code\wuyahe\2.data\260312颏孔分类\260312颏孔分类\3")