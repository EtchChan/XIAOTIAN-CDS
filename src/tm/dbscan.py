###
# python=3.8.5
# version=0.1.0
# workspace_root=$(project_root)
# deprecated, due to too sparse point cloud
###

import pandas as pd
import numpy as np
from sklearn.neighbors import KDTree
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from utilities import read_config, polar_to_rect, rect_to_polar, pd_to_np, np_to_pd

config_file_path = "src/tm/config/final_visual_data_3.yaml"
# 读取配置文件
config = read_config(config_file_path)
print("dbscan configuration:")
print(config)

# 读取数据文件
file_path = f"../materials/finals/{config['file_name']}.xlsx"  # Update this to the correct file path
df = pd.read_excel(file_path)

# 选取需要处理的帧
if config["end_loop"] == -1:
    config["end_loop"] = np.max(df["loop"])
    end_loop_real = -1
else:
    end_loop_real = config["end_loop"]

selected_data = df[
    (df["loop"] >= config["start_loop"]) & (df["loop"] <= config["end_loop"])
]
# 添加新列
selected_data.loc[:, "id"] = 0
selected_data.loc[:, "v_x"] = 0
selected_data.loc[:, "v_y"] = 0
selected_data.loc[:, "v_z"] = 0
selected_data.loc[:, "x_k"] = 0
selected_data.loc[:, "y_k"] = 0
selected_data.loc[:, "z_k"] = 0
df = np_to_pd(polar_to_rect(pd_to_np(selected_data)))

# 构建 KD 树，使用 (x, y, z) 坐标来寻找邻居
tree = KDTree(df[["x", "y", "z"]].values)

# 定义邻域搜索半径（如 0.5 米）
radius = 100

# 定义特征列表
features = []

# 遍历每个点，计算邻居分布特征
for i in range(len(df)):
    # 找到每个点的邻居索引
    indices = tree.query_radius(df[["x", "y", "z"]].values[i : i + 1], r=radius)[0]

    # 若没有邻居点则跳过
    if len(indices) <= 1:
        continue

    # 提取邻居的时间和空间坐标
    neighbors = df.iloc[indices]
    time_mean = neighbors["time"].mean()
    spatial_data = neighbors[["x", "y", "z"]].values
    spatial_mean = neighbors[["x", "y", "z"]].mean().values
    spatial_std = neighbors[["x", "y", "z"]].std().values
    neighbor_count = len(indices)  # 邻近点数量

    # 使用 PCA 提取第一个主方向
    if len(spatial_data) > 1:  # 至少需要两个点才能计算 PCA
        pca = PCA(n_components=1)  # 只提取第一个主成分
        pca.fit(spatial_data)
        principal_component = pca.components_[0]  # 获取第一个主成分
    else:
        principal_component = np.zeros(3)  # 如果只有一个点，主成分为零向量

    # 将特征存入列表
    features.append([time_mean, *principal_component, neighbor_count])

# 转换为 DataFrame
# features_df = pd.DataFrame(
#     features,
#     columns=[
#         "time_mean",
#         "x_mean",
#         "y_mean",
#         "z_mean",
#         "x_std",
#         "y_std",
#         "z_std",
#         "n",
#     ],
# )
features_df = pd.DataFrame(
    features, columns=["time_mean", "pca1", "pca2", "pca3", "neighbor_count"]
)
features_df.fillna(features_df.mean(), inplace=True)  # 用均值填充
# 归一化特征
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_df)

# # 检查是否有 NaN 值
# nan_indices = np.isnan(features_scaled).any(axis=1)
# # 找到所有含有 NaN 的行
# nan_rows = features_scaled[nan_indices]
# print("含有 NaN 的行：", nan_rows)

# 使用 DBSCAN 聚类
dbscan = DBSCAN(eps=0.3, min_samples=3)  # 可调整eps和min_samples参数
labels = dbscan.fit_predict(features_scaled)

# 将聚类结果添加到原始数据
df["id"] = -1  # 默认标签为 -1
df.loc[features_df.index, "id"] = labels

# 输出每个聚类的大小
cluster_sizes = df["id"].value_counts()
print(cluster_sizes)

# 过滤得到有效的聚类点集（排除噪声点）
valid_clusters = df[df["id"] != -1]

# 查看结果
print(valid_clusters)

valid_clusters.to_excel(
    f"output/finals/{config['file_name']}_{config['start_loop']}_{end_loop_real}_dbscan.xlsx",
    index=False,
)
