import numpy as np
import matplotlib.pyplot as plt

# 定义SiLU激活函数
def silu(x):
    return x / (1 + np.exp(-x))

# 生成输入数据
x = np.linspace(-10, 10, 400)
y = silu(x)

# 绘制图形
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='SiLU', color='r')
plt.title('SiLU Activation Function')
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend()
plt.grid(True)
plt.show()
# plt.savefig('./images/silu.png')