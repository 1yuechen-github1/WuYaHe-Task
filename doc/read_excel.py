# import pandas as pd
import pandas as pd
# import pandas as pd
import os
import shutil

def read_excel(path):
    df = pd.read_excel(path)
    return df

def classify_img(df,img_path):
    path = r'C:\Users\yuechen\Desktop\center'
    for index, row in df.iterrows():
        classify_img = str((row['分类']))
        file = row['ct编号']
        
        file = int(file)
        file = str.format('{:03d}', file) + '_中间截面.png'
        print('file:',file)
        os.makedirs(os.path.join(path, classify_img), exist_ok=True)
        shutil.copy(os.path.join(img_path, file), os.path.join(path, classify_img,file))
        # print(classify_img)

img_path = r'\\Desktop-76khoer\d\1.CY-SPACE\WuYaHe\0112NewCt\2.12SeverePeriodontitis\1.General Hospital\NII\img\crop'
df = read_excel(r'\\Desktop-76khoer\d\1.CY-SPACE\WuYaHe\0112NewCt\2.12SeverePeriodontitis\1.General Hospital\NII\img\excel\260304两批总院正中截面分类1.xlsx')
classify_img(df,img_path)