import numpy as np
import matplotlib.pyplot as plt

# 定义Tanh激活函数
def tanh(x):
    return np.tanh(x)

# 生成输入数据
x = np.linspace(-10, 10, 400)
y = tanh(x)

# 绘制图形
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='Tanh', color='r')
plt.title('Tanh Activation Function')
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend()
plt.grid(True)
plt.show()
# plt.savefig('./images/tanh.png')