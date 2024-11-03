from PIL import Image
import numpy as np

# 打开图片
img = Image.open('food11/training/0_3.jpg')

# 转换为 NumPy 数组
img_array = np.array(img)

# 计算均值和标准差
mean = np.mean(img_array, axis=(0, 1))  # 计算 RGB 每个通道的均值
std = np.std(img_array, axis=(0, 1))    # 计算 RGB 每个通道的标准差

print(f'Mean: {mean}, Std: {std}')

