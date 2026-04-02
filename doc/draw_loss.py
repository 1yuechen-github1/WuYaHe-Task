import os
import matplotlib.pyplot as plt

def draw_loss(path):
    for file in os.listdir(path):
        if file.endswith('loss.txt'):
            with open(os.path.join(path, file), 'r') as f:
                lines = f.readlines()
                loss_values = []
                for line in lines:
                    line = line.strip()
                    if line:
                        try:
                            loss = float(line)
                            loss_values.append(loss)
                        except:
                            pass
                
                if loss_values:
                    plt.figure(figsize=(10, 6))
                    plt.plot(range(len(loss_values)), loss_values, 'b-', linewidth=2)
                    plt.title('Loss Curve')
                    plt.xlabel('Epoch')
                    plt.ylabel('Loss')
                    plt.grid(True)
                    
                    output_path = os.path.join(r'C:\yuechen\code\wuyahe\3.实验结果\0316\loss1', file.replace('loss.txt', 'loss.png'))
                    plt.savefig(output_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f'Loss curve saved to {output_path}')

# 示例用法
if __name__ == '__main__':
    # 替换为你的loss.txt文件所在目录
    draw_loss(r'C:\yuechen\code\wuyahe\3.实验结果\0316\loss')
