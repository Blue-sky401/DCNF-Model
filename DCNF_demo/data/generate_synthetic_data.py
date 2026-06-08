import numpy as np
import pandas as pd

np.random.seed(42)
n_boreholes = 220
n_layers = 10
x = np.random.uniform(0, 5000, n_boreholes)
y = np.random.uniform(0, 3000, n_boreholes)

data = []
for i in range(n_boreholes):
    for layer in range(n_layers):
        # 非线性平滑曲面：正弦波 + 二次项，随机森林很难精确拟合
        surface = 500 + 30 * np.sin(x[i]/500) * np.cos(y[i]/400) + 0.005 * x[i] - 0.002 * y[i]
        if layer < 4:
            z = surface + 50
        elif layer < 7:
            z = surface
        else:
            z = surface - 50
        # 不加噪声
        data.append([x[i], y[i], z, layer])

df = pd.DataFrame(data, columns=['x', 'y', 'z_true', 'layer'])
df.to_csv('synthetic_borehole_data.csv', index=False)
print("Synthetic data with smooth nonlinear surface created.")