import numpy as np
import matplotlib.pyplot as plt

# 定义Softmax激活函数
def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=0)

# 生成输入数据
x = np.linspace(-2, 2, 400)
X = np.vstack([x, x + 1, x - 1])

# 计算Softmax输出
Y = softmax(X)

# 绘制图形
plt.figure(figsize=(10, 6))
for i in range(Y.shape[0]):
    plt.plot(x, Y[i], label=f'Softmax output {i+1}')

plt.title('Softmax Activation Function')
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend()
plt.grid(True)
plt.show()
# plt.savefig('./images/softmax.png')