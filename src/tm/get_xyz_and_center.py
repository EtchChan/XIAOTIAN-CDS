import pandas as pd
import numpy as np

def calculate_cartesian_coordinates(df):
    # 提取斜距、方位角和俯仰角数据
    r = df["斜距(m)"]
    theta = np.deg2rad(df["方位角（°）"])  # 将角度转换为弧度
    phi = np.deg2rad(df["俯仰角（°）"])  # 将角度转换为弧度

    # 计算直角坐标系坐标
    df["x"] = r * np.cos(theta) * np.cos(phi)
    df["y"] = r * np.sin(theta) * np.cos(phi)
    df["z"] = r * np.sin(phi)

    return df

def calculate_mean_coordinates(df):
    # 计算同一圈内的直角坐标系均值
    mean_coords = df.groupby("圈数")[["x", "y", "z"]].mean().reset_index()
    mean_coords.columns = ["圈数", "mean_x", "mean_y", "mean_z"]

    return mean_coords

def main(input_file, output_file):
    # 读取 Excel 文件
    df = pd.read_excel(input_file)

    # 计算直角坐标系坐标
    df = calculate_cartesian_coordinates(df)

    # 计算同一圈内的直角坐标系均值
    mean_coords = calculate_mean_coordinates(df)

    # 将均值数据合并到原始数据中
    df = pd.merge(df, mean_coords, on="圈数", how="left")

    # 将结果写入新的 Excel 文件
    df.to_excel(output_file, index=False)

    print(f"处理完成，结果已保存到 {output_file}")

if __name__ == "__main__":
    file_path = "./output/filtered/kalman_results_dataset1_centers.xlsx"
    input_file = file_path  # 输入的 Excel 文件路径
    output_file = "./output/filtered/processed_data.xlsx"  # 输出的 Excel 文件路径

    main(input_file, output_file)