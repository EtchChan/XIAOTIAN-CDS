###
# python=3.8.5
# version=0.1.0
# workspace_root=$(project_root)
# ############
# Deprecated
# We can not assume that the trajectory is a line
###


import pandas as pd
import numpy as np
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from utilities import read_config, polar_to_rect, rect_to_polar, pd_to_np, np_to_pd

# 读取配置文件
config = read_config("src/tm/config/final_visual_data_1.yaml")
print("ransac configuration:")
print(config)

# 读取数据文件
file_path = f"../materials/finals/{config['file_name']}.xlsx"  # Update this to the correct file path
df = pd.read_excel(file_path)

df = np_to_pd(polar_to_rect(pd_to_np(df[df["r"] > 1500])))
# 选择数据中的 x, y, z 列
X = df[["x", "y", "z"]].values

# 标准化数据以便更稳定的处理
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 使用 PCA 将数据降低到一维，用于初步线性结构提取
pca = PCA(n_components=1)
X_pca = pca.fit_transform(X_scaled)

# 逆变换回原始空间，方便后续拟合
X_approx = pca.inverse_transform(X_pca)

# 计算残差（原始数据与拟合数据的距离）
residuals = np.linalg.norm(X_scaled - X_approx, axis=1)

# 设置残差阈值，用于区分内点和外点
threshold = np.percentile(residuals, 90)  # 可根据需求调整百分比

# 只保留接近直线的点
inliers = residuals < threshold
X_inliers = X[inliers]

# 在内点上使用 RANSAC 来拟合直线模型
# 在内点上使用 RANSAC 来拟合 y = f(x) 和 z = f(x) 的模型
ransac_y = RANSACRegressor(
    min_samples=150,  # 每次拟合模型所需的最小数据点数
    residual_threshold=5,  # 确定数据点是否符合模型的阈值
    max_trials=5000,  # 最大迭代次数
    stop_n_inliers=300,  # 停止迭代的内点数阈值
    stop_probability=0.99,  # 停止迭代的概率阈值
)
ransac_z = RANSACRegressor(
    min_samples=150,  # 每次拟合模型所需的最小数据点数
    residual_threshold=5,  # 确定数据点是否符合模型的阈值
    max_trials=5000,  # 最大迭代次数
    stop_n_inliers=300,  # 停止迭代的内点数阈值
    stop_probability=0.99,  # 停止迭代的概率阈值
)

# 拟合模型
ransac_y.fit(X_inliers[:, [0]], X_inliers[:, 1])  # 拟合 y = f(x)
ransac_z.fit(X_inliers[:, [0]], X_inliers[:, 2])  # 拟合 z = f(x)

# 提取拟合结果：直线模型的系数和截距
slope_y = ransac_y.estimator_.coef_[0]
intercept_y = ransac_y.estimator_.intercept_
slope_z = ransac_z.estimator_.coef_[0]
intercept_z = ransac_z.estimator_.intercept_

# 输出直线方程
print(f"拟合直线方程：y = {slope_y:.3f} * x + {intercept_y:.3f}")
print(f"拟合直线方程：z = {slope_z:.3f} * x + {intercept_z:.3f}")

# 筛选出 RANSAC 识别的内点
inliers_ransac_y = ransac_y.inlier_mask_
inliers_ransac_z = ransac_z.inlier_mask_
final_inliers_y = X_inliers[inliers_ransac_y]
final_inliers_z = X_inliers[inliers_ransac_z]

# 将结果转为 DataFrame 以便进一步分析或可视化
final_inliers_df_y = pd.DataFrame(final_inliers_y, columns=["x", "y", "z"])
final_inliers_df_z = pd.DataFrame(final_inliers_z, columns=["x", "y", "z"])
print(final_inliers_df_y)
print(final_inliers_df_z)

# 参数方程形式
# 选择一个点 r0 (x0, y0, z0)
x0 = np.min(X_inliers[:, 0])  # 选择 x 的最小值作为起点
y0 = slope_y * x0 + intercept_y
z0 = slope_z * x0 + intercept_z

# 方向向量 d (dx, dy, dz)
dx = 1  # 选择任意方向向量在 x 方向上的分量
dy = slope_y  # 方向向量在 y 方向上的分量
dz = slope_z  # 方向向量在 z 方向上的分量

# 输出参数方程
print("参数方程形式：")
print(f"x = {x0} + t * {dx}")
print(f"y = {y0} + t * {dy}")
print(f"z = {z0} + t * {dz}")
