import os


def read_txt(path):
    qianya_path = os.path.join(path, 'qianya')
    # for file in os.listdir(qianya_path):
    file_path = os.path.join(qianya_path, '001', 'len.txt')
    values = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            value = int(line.split(',')[-1]) * 0.3
            values.append(value)
    print(values)
read_txt(r'C:\yuechen\code\wuyahe\1.code\2.data-缩放\screenshot\pca-sum')
