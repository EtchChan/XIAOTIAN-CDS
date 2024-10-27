###
# python=3.8.5
# version=0.1.0
# workspace_root=$(project_root)
###

import pandas as pd
import numpy as np
import yaml

# 常量2pi
PI2 = 2 * np.pi

# 常量颜色表
# Create a color map for each loops
# colors = ["red", "green", "blue", "orange", "purple"]
colors = [
    # "#FF0000",  # Red
    "#00A5FF",  # ?
    "#00FF00",  # Green
    "#0000FF",  # Blue
    "#00FFFF",  # Cyan
    "#FF00FF",  # Magenta
    "#FFFF00",  # Yellow
    "#000000",  # Black
    "#FFFFFF",  # White
    "#808080",  # Gray
    "#FFA500",  # Orange # 10
    "#FFA500",  # Orange
    "#FFA500",  # Orange
    "#FFA500",  # Orange
    "#FFA500",  # Orange
    "#FFA500",  # Orange # 15
]


def read_config(config_file_path):
    """
    读取配置文件，返回处理的文件名、处理的起始帧、处理的最后帧、是否是原始数据、阈值等等。

    参数:
    config_file_path (str): 配置文件的路径

    返回:
    dict: 包含配置内容的字典
    """
    with open(config_file_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return {
        "file_name": config.get("file_name"),
        "data_type": config.get("data_type"),
        "start_loop": config.get("start_loop"),
        "end_loop": config.get("end_loop"),
        "threshold": config.get("threshold"),
    }


# 极坐标到直角坐标转换函数
def polar_to_rect(*args):
    if len(args) == 1:
        args = np.array(args[0]).T
        # (args[1], args[2], args[3]) => (r, theta, phi) -> (x, y, z)
        r, theta, phi = np.copy(args[1]), np.deg2rad(args[2]), np.deg2rad(args[3])
        args[7] = r * np.cos(theta) * np.cos(phi)
        args[8] = r * np.sin(theta) * np.cos(phi)
        args[9] = r * np.sin(phi)
        return args.T
    elif len(args) == 3:
        # (r, theta, phi) -> (x, y, z)
        (r, theta, phi) = args
        theta = np.deg2rad(theta)
        phi = np.deg2rad(phi)
        x = r * np.cos(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.cos(phi)
        z = r * np.sin(phi)
        return x, y, z
    else:
        print(
            "[WARNING] polar_to_rect receives bad input!!/n"
            "Expected input length should be 1 or 3!"
        )
        return args


# 直角坐标到极坐标转换函数
def rect_to_polar(*args):
    if len(args) == 1:
        args = np.array(args[0]).T
        # (args[1], args[2], args[3]) => (x, y, z) -> (r, theta, phi)
        x, y, z = np.copy(args[7]), np.copy(args[8]), np.copy(args[9])
        args[1] = np.sqrt(x**2 + y**2 + z**2)
        args[2] = np.rad2deg(np.mod(np.arctan2(y, x) + PI2, PI2))
        args[3] = np.rad2deg(np.mod(np.arctan2(z, np.sqrt(x**2 + y**2)) + PI2, PI2))
        return args.T
    elif len(args) == 3:
        # (x, y, z) -> (r, theta, phi)
        x, y, z = args
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.rad2deg(np.arctan2(y, x))
        phi = np.rad2deg(np.arctan2(z, np.sqrt(x**2 + y**2)))
        return r, theta, phi
    else:
        print(
            "[WARNING] rect_to_polar receives bad input!!\n"
            "Expected input length should be 1 or 3!"
        )
        return args


# panda data_frame 到 numpy array 转换函数
def pd_to_np(df):
    # return is np:
    # t r theta phi v_r ...
    # t r theta phi v_r ...
    # . .   .    .   .
    # . .   .    .   .

    # extract the data
    t = df["time"]
    r = df["r"]
    theta = df["azimuth"]
    phi = df["pitch"]
    v_r = df["v_r"]
    loop_idx = df["loop"]
    traj_id = df["id"]
    x = df["x"]
    y = df["y"]
    z = df["z"]
    v_x = df["v_x"]
    v_y = df["v_y"]
    v_z = df["v_z"]
    x_k = df["x_k"]
    y_k = df["y_k"]
    z_k = df["z_k"]

    # 封装并返回
    return np.vstack(
        (
            t,
            r,
            theta,
            phi,
            v_r,
            loop_idx,
            traj_id,
            x,
            y,
            z,
            v_x,
            v_y,
            v_z,
            x_k,
            y_k,
            z_k,
        )
    ).T


# numpy array 到 panda data_frame 转换函数
def np_to_pd(np_array):

    t = np_array[:, 0]
    r = np_array[:, 1]
    theta = np_array[:, 2]
    phi = np_array[:, 3]
    v_r = np_array[:, 4]
    loop_idx = np_array[:, 5]
    traj_id = np_array[:, 6]
    x = np_array[:, 7]
    y = np_array[:, 8]
    z = np_array[:, 9]
    v_x = np_array[:, 10]
    v_y = np_array[:, 11]
    v_z = np_array[:, 12]
    x_k = np_array[:, 13]
    y_k = np_array[:, 14]
    z_k = np_array[:, 15]

    return pd.DataFrame(
        {
            "time": t,
            "r": r,
            "azimuth": theta,
            "pitch": phi,
            "v_r": v_r,
            "loop": loop_idx,
            "id": traj_id,
            "x": x,
            "y": y,
            "z": z,
            "v_x": v_x,
            "v_y": v_y,
            "v_z": v_z,
            "x_k": x_k,
            "y_k": y_k,
            "z_k": z_k,
        }
    )
