import numpy as np
import matplotlib.pyplot as plt

# 定义Swish激活函数
def swish(x):
    return x * (1 / (1 + np.exp(-x)))

# 定义SwiGLU函数
def swiglu(x, W1=1, b1=0):
    return swish(x) * (W1 * x + b1)

# 生成输入数据
x = np.linspace(-5, 5, 400)
y = swiglu(x)

# 绘制图形
plt.figure(figsize=(10, 6))
plt.plot(x, swiglu(x, W1=2), label='SwiGLU, W1=2', linestyle='--', color='r')
plt.plot(x, swiglu(x), label='SwiGLU W1=1', color='b')
plt.plot(x, swiglu(x, W1=0.5), label='SwiGLU, W1=0.5', linestyle='-.', color='g')
plt.plot(x, swiglu(x, W1=0.1), label='SwiGLU, W1=0.1', linestyle=':', color='m')
plt.title('SwiGLU Activation Function')
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend()
plt.grid(True)
plt.show()
# plt.savefig('./images/swiglu.png')