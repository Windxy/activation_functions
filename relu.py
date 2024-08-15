import numpy as np
import matplotlib.pyplot as plt

# 定义ReLU激活函数
def relu(x):
    return np.maximum(0, x)

# 生成输入数据
x = np.linspace(-10, 10, 400)
y = relu(x)

# 绘制图形
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='ReLU', color='r')
plt.title('ReLU Activation Function')
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend()
plt.grid(True)
plt.show()
# plt.savefig('./images/relu.png')