import numpy as np
import matplotlib.pyplot as plt

# 定义GELU激活函数
def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

# 生成输入数据
x = np.linspace(-10, 10, 400)
y = gelu(x)

# 绘制图形
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='GELU', color='r')
plt.title('GELU Activation Function')
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend()
plt.grid(True)
plt.show()
# plt.savefig('./images/gelu.png')