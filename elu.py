import numpy as np
import matplotlib.pyplot as plt

# 定义ELU激活函数
def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

# 生成输入数据
x = np.linspace(-10, 10, 400)
y = elu(x)

# 绘制图形
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='ELU', color='r')
plt.title('ELU Activation Function')
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('./images/elu.png')