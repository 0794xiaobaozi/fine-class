import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import os

# Step 1: Load Data from Excel
file_paths = [
    r"F:\Junior\mech\five\five\1\Book1.xlsx",
    r"F:\Junior\mech\five\five\3\Book1.xlsx",
    ]
# 6.96*2.166667
# 4.1
# 5.4*2.63
# 4.79*2.913333
# 7.0
# 10.02
# 7.0
# 9.3

# 用户设置的转换因子
# load_conversion_factors = [6.96*2.166667, 5.4*2.63, 4.79*2.913333]  # 每组数据的载荷转换因子
load_conversion_factors = [6.96*2.166667, 5.4*2.63]  # 每组数据的载荷转换因子
strain_conversion_factors = [7.0, 7.0]  # 每组数据的变形转换因子
# strain_conversion_factors = [7.0, 7.0, 9.3]  # 每组数据的变形转换因子


# 检查文件是否存在
for file_path in file_paths:
    if not os.path.exists(file_path):
        print(f"文件路径不存在，请检查路径是否正确：{file_path}")
        exit()

# Step 2: 读取每组数据并进行处理
group_data = []

for i, file_path in enumerate(file_paths):
    data = pd.read_excel(file_path, sheet_name=3, header=0)  # 读取Excel的第一个工作表

    # 提取载荷和当径列（假设第2列是载荷，第3列是当径），并进行转换
    load_column = data.iloc[:, 1] / load_conversion_factors[i]  # 转换为应力，单位 MPa
    strain_column = data.iloc[:, 2] / (strain_conversion_factors[i] * 10)  # 转换为无单位真实变形

    # 创建 DataFrame，合并应力和真实变形数据
    load_strain_data = pd.DataFrame({
        "Stress (MPa)": load_column,
        "True Strain": strain_column
    }).dropna()

    group_data.append(load_strain_data)

# Step 3: 绘制三组数据的曲线
plt.figure(figsize=(12, 8))
colors = ['blue', 'green', 'orange']
for i, data in enumerate(group_data):
    plt.plot(data['True Strain'], data['Stress (MPa)'], label=f'Group {i+1}', color=colors[i], linewidth=1)

# Step 4: 计算均值曲线和误差
combined_strain = np.linspace(0, min([data['True Strain'].max() for data in group_data]), 500)
combined_stress = []

for strain_value in combined_strain:
    stresses = [np.interp(strain_value, data['True Strain'], data['Stress (MPa)']) for data in group_data]
    combined_stress.append(stresses)

combined_stress = np.array(combined_stress)
mean_stress = combined_stress.mean(axis=1)
std_stress = combined_stress.std(axis=1)

# 绘制均值曲线和误差线
plt.plot(combined_strain, mean_stress, label='Mean Curve', color='red', linewidth=2)
plt.fill_between(combined_strain, mean_stress - std_stress, mean_stress + std_stress, color='red', alpha=0.2, label='Error Range')

# Step 5: 确定线性区域的自变量范围
linear_start = 0.17
linear_end = 0.28

# 获取均值曲线的线性区域并计算斜率
linear_region_indices = (combined_strain >= linear_start) & (combined_strain <= linear_end)
linear_strain = combined_strain[linear_region_indices]
linear_stress = mean_stress[linear_region_indices]

lin_reg = LinearRegression()
lin_reg.fit(linear_strain.reshape(-1, 1), linear_stress)
linear_slope = lin_reg.coef_[0]

# 标注均值曲线的线性区域
plt.axvspan(linear_start, linear_end, color='yellow', alpha=0.3, label='Linear Region (Mean Curve)')
midpoint_strain = (linear_start + linear_end) / 2
midpoint_stress = lin_reg.predict([[midpoint_strain]])[0]
plt.text(midpoint_strain, midpoint_stress, f'Slope: {linear_slope:.2f} MPa/unit', color='red', ha='center', fontsize=10)

# 标注线性区域的开始点和结束点
plt.scatter(linear_start, lin_reg.predict([[linear_start]])[0], color='red', s=100)
plt.text(linear_start, lin_reg.predict([[linear_start]])[0], f'Start ({linear_start:.2f})', color='red', ha='left')
plt.scatter(linear_end, lin_reg.predict([[linear_end]])[0], color='red', s=100)
plt.text(linear_end, lin_reg.predict([[linear_end]])[0], f'End ({linear_end:.2f})', color='red', ha='right')

# 图表设置
plt.xlabel("True Strain")
plt.ylabel("Stress (MPa)")
plt.title("Stress-Strain Curves with Mean Curve and Linear Region")
plt.legend(loc="upper left")
plt.grid(True)
plt.show()
