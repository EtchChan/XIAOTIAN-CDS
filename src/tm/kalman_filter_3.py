###
# python=3.8.5
# version=0.1.0
# workspace_root=$(project_root)
###

import pandas as pd
import numpy as np
from pykalman import KalmanFilter
import time

from utilities import read_config, polar_to_rect, rect_to_polar, pd_to_np, np_to_pd

################
# 全局包配置区
################
# 警告！这里全局禁用链式赋值警告，因为在增加新列时会给4个新列都赋值一样的np.nan，因此pandas会直接复制列，从而产生SettingWithCopyWarning警告
# 但是官方指出在3.0版本之后已经默认开启Copy-on-Write模式，因此在普通的赋值的时候也可能会出现上述警报。
# 虽然始终检查每一个可能的赋值语句有助于减少代码出错的可能性，但是本程序代码量较少，这种繁杂的警告可能会有负面效果，因此在此处将其禁用
# 在遇到可能的赋值错误或者发现数据被意外修改时，应启用该警告以查看可能出错的区域
pd.set_option("mode.chained_assignment", None)

################
# 全局变量区
################

#
current_trajectory_id = 0

# 运动学转移矩阵，可修改，但是应当符合transition matrix的基本要求
# (x, y, z, v_x, v_y, v_z)
transition_matrix = np.array(
    [
        [1, 0, 0, 3, 0, 0],  # 3s 一圈
        [0, 1, 0, 0, 3, 0],
        [0, 0, 1, 0, 0, 3],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
    ]
)


################
# 函数区
################


def timeit(func):
    """
    装饰器函数，用于记录函数的执行时间。
    """

    def wrapper(*args, **kwargs):
        start_time = time.time()  # 记录开始时间
        result = func(*args, **kwargs)
        end_time = time.time()  # 记录结束时间
        execution_time = end_time - start_time  # 计算运行时间
        print(f"{func.__name__} 运行时间：{execution_time:.4f} s")
        return result

    return wrapper


@timeit
def pre_process(file_path):
    # 读取配置文件
    config = read_config(file_path)
    print("kalman_filter v3 configuration:")
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
    selected_data.loc[:, "轨迹ID"] = np.nan
    selected_data.loc[:, "v_x"] = np.nan
    selected_data.loc[:, "v_y"] = np.nan
    selected_data.loc[:, "v_z"] = np.nan
    selected_data.loc[:, "x_k"] = np.nan
    selected_data.loc[:, "y_k"] = np.nan
    selected_data.loc[:, "z_k"] = np.nan
    return selected_data, config, end_loop_real


# 初始化卡尔曼滤波器，状态包括位置和速度
def initialize_kalman_filter():
    global transition_matrix

    # 观测无转换
    observation_matrix = np.eye(3, 6)

    return KalmanFilter(
        transition_matrices=transition_matrix,
        observation_matrices=observation_matrix,
        initial_state_mean=np.zeros(6),
        initial_state_covariance=1.0 * np.eye(6),
        transition_covariance=0.1 * np.eye(6),
        observation_covariance=0.1 * np.eye(3),
    )


# 普通的运动学预测
def predict(states):
    global transition_matrix
    predicts = states.copy()
    predicts.iloc[:, 7:10] = ((transition_matrix @ predicts.T.iloc[7:13]).iloc[:3]).T
    return predicts


# TODO match 和 kalman 之间可以设计一个更好的结构，用来传递匹配的结果
def match(currents, predicts, config):
    global current_trajectory_id
    threshold = config["threshold"]
    for i in range(len(currents)):
        # 将当前的每一个点与上一圈的普通运动学预测相比较
        diff_np = currents.iloc[i, 7:10].to_numpy() - predicts.iloc[:, 7:10].to_numpy()
        distances = np.linalg.norm(
            # diff_np,
            diff_np[:, :2],  # 只考虑 x, y 的距离
            axis=1,
        )
        if (
            threshold[0] < np.min(distances) < threshold[1]
            ## （以下均为特殊处理)
            and currents.iloc[i, 8] < 0
            # and diff_np < 50
            # and currents.iloc[i, 1] > threshold[2]  # 斜距 r_min 以上的点才匹配
            and currents.iloc[i, 9] > threshold[3]  # z轴大于 z_min 以上才匹配
        ):
            # 如果匹配到了，则更新当前点的轨迹ID
            temp_idx = np.argmin(distances)
            currents.iloc[i, 6] = predicts.iloc[temp_idx, 6]
            # 更新直角坐标系速度
            currents.iloc[i, 10:13] = (
                currents.iloc[i, 7:10] - predicts.iloc[temp_idx, 7:10]
            ) / (currents.iloc[i, 0] - predicts.iloc[temp_idx, 0])
        else:
            # 没有匹配到，说明当前的点是新出现的，添加新的ID
            currents.iloc[i, 6] = current_trajectory_id
            current_trajectory_id += 1
    # print("---- after match ----")
    # print(currents)
    return currents


# 选取普通运动学预测中的邻近作为观测，更新kalman状态
# TODO 当前版本的多目标更新采用相同的协方差矩阵，后续版本应当考虑每一个目标都维护一个协方差矩阵
def kalman_update_2(kf, obs, previous_loop, state_covariance=1.0 * np.eye(6)):

    for i in range(len(obs)):
        if not np.isnan(obs.iloc[i, 10]):
            # 如果线速度不为nan，说明有匹配
            # 通过轨迹ID来查找对应的pred
            pred = previous_loop[previous_loop.iloc[:, 6] == obs.iloc[i, 6]].iloc[0]
            temp_obs, state_covariance = kf.filter_update(
                filtered_state_mean=np.hstack([pred.iloc[7:13]]),
                filtered_state_covariance=state_covariance,
                observation=obs.iloc[i, 7:10],
            )
            obs.iloc[i, 13:16] = temp_obs[:3]
        else:
            # 如果没有匹配上，说明是新出现的点，直接更新径向速度，状态不变
            obs.iloc[i] = update_vel(obs.iloc[i])

    return obs, state_covariance


# TODO 修改此处，要进行正确的第二帧之后的预测
def update_vel(*args):
    data = args[0]
    # 提取重复计算的部分
    v_r = data["v_r"]
    cos_theta = np.cos(np.deg2rad(data["azimuth"]))
    sin_theta = np.sin(np.deg2rad(data["azimuth"]))
    cos_phi = np.cos(np.deg2rad(data["pitch"]))
    sin_phi = np.sin(np.deg2rad(data["pitch"]))
    data["v_x"] = v_r * cos_theta * cos_phi
    data["v_y"] = v_r * sin_theta * cos_phi
    data["v_z"] = v_r * sin_phi
    return data


## TODO 如果点实在是太少的话，可以考虑把 trajectory ID 设置为 -1，表示噪声
@timeit
def process(data, config):
    global current_trajectory_id
    global transition_matrix

    # 初始化滤波器
    kf = initialize_kalman_filter()

    # 初始化状态协方差矩阵
    state_covariance = 1.0 * np.eye(6)

    # 加入转换后的坐标
    #  0  1    2     3    4     5     6        7  8  9  10   11   12
    # (t, r, theta, phi, v_r, lp_idx, traj_id, x, y, z, v_x, v_y, v_z)
    data = np_to_pd(polar_to_rect(pd_to_np(data)))

    # 初始帧处理，获取初始帧预测，预测根据为径向速度
    first_loop_data = data[data["loop"] == config["start_loop"]]
    # 增加轨迹ID列，用于存储轨迹ID
    first_loop_data.loc[:, "id"] = range(
        current_trajectory_id, current_trajectory_id + len(first_loop_data)
    )
    current_trajectory_id += len(first_loop_data)

    # 更新预测中的速度
    first_loop_data = update_vel(first_loop_data)
    previous_states = first_loop_data.copy()
    #  0  1  2  3   4     5     6         7    8    9
    # (t, x, y, z, v_r, lp_idx, traj_id, v_x, v_y, v_z)

    # 累计记录数据
    accumulate = [first_loop_data]

    # 执行普通运动学预测，这里的previous_predicts是相对于下一帧（当前帧）来说的
    previous_predicts = predict(first_loop_data)

    for loop_index in range(config["start_loop"] + 1, config["end_loop"] + 1):
        current_loop_data = data[data["loop"] == loop_index]
        # print(current_loop_data_rect)
        # 匹配当前帧所有点与上一帧预测，这里会加入匹配到的点的速度
        current_after_match = match(current_loop_data, previous_predicts, config)
        # 通过匹配结果执行kalman滤波器更新, 匹配的结果作为卡尔曼的观测
        # 同时，在这个版本的卡尔曼中，一同完成了非匹配点的速度更新

        # 直接取消kalman试试
        current_after_kalman, state_covariance = kalman_update_2(
            kf,
            current_after_match,  # obs
            previous_states,  # pre
            state_covariance,
        )

        # 加入累计数组
        accumulate.append(current_after_kalman)

        # 更新上一帧的状态
        previous_states = current_after_kalman.copy()

        # 继续执行普通运动学预测，为下一圈做好准备
        previous_predicts = predict(previous_states)

    return accumulate


@timeit
def post_process(data, config, end_loop_real):
    data.to_excel(
        f"output/finals/{config['file_name']}_{config['start_loop']}_{end_loop_real}_{config['threshold']}.xlsx",
        index=False,
    )


@timeit
def get_center_trajectory(data, config, end_loop_real):
    # 获取中心航迹
    # 提取最多的 15 个 id
    top_15_ids = data["id"].value_counts().head(15).index

    # 过滤数据
    filtered_data = data[data["id"].isin(top_15_ids)]

    # 计算每个 loop_index 中 x, y, z 的均值
    center_coords = (
        filtered_data.groupby("loop")[["x", "y", "z", "time"]].mean().reset_index()
    )

    print("每个 loop 中 x, y, z 的和时间的均值:\n", center_coords)

    # 计算斜距、方位角和高低角
    center_coords["群中心斜距(m)"] = np.sqrt(
        center_coords["x"] ** 2 + center_coords["y"] ** 2 + center_coords["z"] ** 2
    )
    center_coords["群中心方位角(°)"] = np.degrees(
        np.arctan2(center_coords["y"], center_coords["x"])
    )
    center_coords["群中心高低角(°)"] = np.degrees(
        np.arctan2(
            center_coords["z"],
            np.sqrt(center_coords["x"] ** 2 + center_coords["y"] ** 2),
        )
    )

    # 计算每个 loop_index 的群规模
    group_size = filtered_data.groupby("loop")["id"].count().reset_index()
    group_size.columns = ["loop", "群规模"]

    # 合并数据
    result = pd.merge(center_coords, group_size, on="loop")

    # 添加圈数列
    result["圈数"] = result["loop"]

    # 选择并重命名列
    result = result[
        [
            "time",
            "群中心斜距(m)",
            "群中心方位角(°)",
            "群中心高低角(°)",
            "x",
            "y",
            "z",
            "群规模",
            "圈数",
        ]
    ]
    result.columns = [
        "时间(s)",
        "群中心斜距(m)",
        "群中心方位角(°)",
        "群中心高低角(°)",
        "群中心X坐标(m)",
        "群中心Y坐标(m)",
        "群中心Z坐标(m)",
        "群规模",
        "圈数",
    ]

    print(result)

    result.to_excel(
        f"output/finals/{config['file_name']}_{config['start_loop']}_{end_loop_real}_{config['threshold']}_center_trajectory.xlsx",
        index=False,
    )


# TODO 增加轨迹权重，当轨迹越长，则减小最小阈值，增加最大阈值
# TODO 可以考虑增加速度滤波，大部分噪声的速度都很小，可以通过速度的大小来过滤噪声
# TODO 可以考虑容忍一两帧的阈值外，否则这个强阈值要求太难了
# TODO 考虑当前点的预测点搜索只在邻近的斜距、方位角、俯仰角范围内进行，这样可以减少计算量
if __name__ == "__main__":

    config_file_path = "src/tm/config/final_visual_data_3.yaml"
    selected_data, config, end_loop_real = pre_process(config_file_path)

    accumulate = process(selected_data, config)
    output_data = pd.concat(accumulate, ignore_index=True)

    # 存储为 Excel 文件
    post_process(output_data, config, end_loop_real)

    # # 得到中心航迹
    get_center_trajectory(output_data, config, end_loop_real)

    print("完成！")
