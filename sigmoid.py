import numpy as np
import matplotlib.pyplot as plt

# 定义Sigmoid激活函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 生成输入数据
x = np.linspace(-10, 10, 400)
y = sigmoid(x)

# 绘制图形
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='Sigmoid', color='r')
plt.title('Sigmoid Activation Function')
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend()
plt.grid(True)
plt.show()
# plt.savefig('./images/sigmoid.png')