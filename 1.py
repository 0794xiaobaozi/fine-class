import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.linear_model import LinearRegression
import os

# Step 1: Load Data from Excel
file_path = r"F:\Junior\mech\five\five\2 failed\Book1.xlsx"  # 请替换为您的实际文件路径

# 用户设置的转换因子
load_conversion_factor = 4.1*2.883333  # 将载荷转换为应力的因子，例如1000
strain_conversion_factor = 10.02  # 将变形转换为无单位真实变形的因子，例如100

# 检查文件是否存在
if not os.path.exists(file_path):
    print("文件路径不存在，请检查路径是否正确。")
else:
    # 读取Excel的第4个工作表，第一行作为表头
    data = pd.read_excel(file_path, sheet_name=3, header=0)

    # 提取载荷和当径列（假设第2列是载荷，第3列是当径），并进行转换
    load_column = data.iloc[:, 1] / load_conversion_factor  # 转换为应力，单位 MPa
    strain_column = data.iloc[:, 2] / (strain_conversion_factor * 10)  # 转换为无单位真实变形

    # 创建 DataFrame，合并应力和真实变形数据
    load_strain_data = pd.DataFrame({
        "Stress (MPa)": load_column,
        "True Strain": strain_column
    }).dropna()

    # 平滑处理
    load_strain_data['Smoothed_Stress'] = savgol_filter(load_strain_data['Stress (MPa)'], window_length=11, polyorder=2)

    # 定义您指定的区间范围
    heel_start, heel_end = 0, 19.856 / (strain_conversion_factor * 10)       # Heel 区域的真实变形区间
    linear_start, linear_end = 19.856 / (strain_conversion_factor * 10), 27.147 / (strain_conversion_factor * 10)  # 线性区域的真实变形区间
    failure_start = 51.315 / (strain_conversion_factor * 10)                 # 失效区域的起始真实变形
    failure_offset = 0.3 / (strain_conversion_factor * 10)                   # 用户设定的失效偏移量

    # Step 2: 获取线性区域的数据并计算斜率
    linear_region_data = load_strain_data[(load_strain_data['True Strain'] >= linear_start) & (load_strain_data['True Strain'] <= linear_end)]
    lin_reg = LinearRegression()
    lin_reg.fit(linear_region_data['True Strain'].values.reshape(-1, 1), linear_region_data['Stress (MPa)'])
    linear_slope = lin_reg.coef_[0]  # 线性区域的斜率

    # Step 3: 计算失效点的应变量值（自变量值为 failure_start - failure_offset）
    adjusted_failure_strain = failure_start - failure_offset
    failure_index = load_strain_data[load_strain_data['True Strain'] >= adjusted_failure_strain].index[0]
    failure_strain = load_strain_data['True Strain'].iloc[failure_index]
    failure_stress = load_strain_data['Stress (MPa)'].iloc[failure_index]

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(load_strain_data['True Strain'], load_strain_data['Stress (MPa)'], label='Original Data', color='blue')
    plt.plot(load_strain_data['True Strain'], load_strain_data['Smoothed_Stress'], label='Smoothed Data', linestyle='--', color='gray')

    # 区间标注
    plt.axvspan(heel_start, heel_end, color='orange', alpha=0.3, label="Heel Region")
    plt.axvspan(linear_start, linear_end, color='green', alpha=0.3, label="Linear Region")
    plt.axvspan(linear_end, failure_start, color='red', alpha=0.3, label="Failure Region")

    # 标注线性区域的斜率
    midpoint_strain = (linear_start + linear_end) / 2
    midpoint_stress = lin_reg.predict([[midpoint_strain]])[0]
    plt.text(midpoint_strain, midpoint_stress, f"Slope: {linear_slope:.2f} MPa/unit", color="green", ha='center', fontsize=10)

    # 标注线性区域的开始点和结束点
    plt.scatter(linear_start, lin_reg.predict([[linear_start]])[0], color='green', s=100)
    plt.text(linear_start, lin_reg.predict([[linear_start]])[0], f"Start ({linear_start:.2f})", color="green", ha='left')
    plt.scatter(linear_end, lin_reg.predict([[linear_end]])[0], color='green', s=100)
    plt.text(linear_end, lin_reg.predict([[linear_end]])[0], f"End ({linear_end:.2f})", color="green", ha='right')

    # # 标注失效点
    plt.scatter(failure_strain, failure_stress, color='red', s=100)
    plt.text(failure_strain, failure_stress, f"({failure_strain:.2f}, {failure_stress:.2f} MPa)", color="red", ha='left')

    # 图表设置：确保原点 (0,0) 且图示框在左上
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.xlabel("Strain")
    plt.ylabel("Stress (MPa)")
    plt.title("Stress-Strain Curve with Labeled Regions and Linear Region Slope")
    plt.legend(loc="upper left")
    plt.show()