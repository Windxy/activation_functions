import numpy as np
import matplotlib.pyplot as plt

# 定义Leaky ReLU激活函数, PRelu激活函数同理，但PRelu激活函数的alpha是一个可学习的参数
def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

# 生成输入数据
x = np.linspace(-10, 10, 400)
y = leaky_relu(x)

# 绘制图形
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='Leaky ReLU', color='r')
plt.title('Leaky ReLU Activation Function')
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend()
plt.grid(True)
plt.show()
# plt.savefig('./images/leaky_relu.png')