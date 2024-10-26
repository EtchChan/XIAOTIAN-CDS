###
# python=3.8.5
# version=0.1.0
# workspace_root=$(project_root)
###

import pandas as pd
import numpy as np
from pykalman import KalmanFilter
import time

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

# 处理的开始圈和最后圈，可修改，但是应当大于0，并且可以超过数据中的最大圈数，若超出最大圈数，则代表所有圈数均进行处理
first_loop_index = 1
total_loops = 309

# 处理文件名
file_name = "data_3.xlsx"

# 匹配阈值，可修改，但是应当符合匹配的基本要求
threshold_top = (10, 150)

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

# 轨迹id，可修改其初始值，但意义不大
current_trajectory_id = 0

PI2 = 2 * np.pi

################
# 函数区
################


# 极坐标到直角坐标转换函数
def polar_to_rect(*args):
    if len(args) == 1:
        args = np.array(args[0]).T
        # (args[1], args[2], args[3]) => (r, theta, phi) -> (x, y, z)
        r, theta, phi = np.copy(args[1]), np.copy(args[2]), np.copy(args[3])
        args[1] = r * np.cos(theta) * np.cos(phi)
        args[2] = r * np.sin(theta) * np.cos(phi)
        args[3] = r * np.sin(phi)
        return args.T
    elif len(args) == 3:
        # (r, theta, phi) -> (x, y, z)
        (r, theta, phi) = args
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
        x, y, z = np.copy(args[1]), np.copy(args[2]), np.copy(args[3])
        args[1] = np.sqrt(x**2 + y**2 + z**2)
        args[2] = np.mod(np.arctan2(y, x) + PI2, PI2)
        args[3] = np.mod(np.arctan2(z, np.sqrt(x**2 + y**2)) + PI2, PI2)
        return args.T
    elif len(args) == 3:
        # (x, y, z) -> (r, theta, phi)
        x, y, z = args
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arctan2(y, x)
        phi = np.arctan2(z, np.sqrt(x**2 + y**2))
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
    # r theta phi v_r
    # r theta phi v_r
    # . .     .   .
    # . .     .   .

    # extract the data
    t = df["时间(s)"]
    r = df["斜距(m)"]
    theta = np.deg2rad(df["方位角（°）"])
    phi = np.deg2rad(df["俯仰角（°）"])
    v_r = df["径向速度（m/s）"]
    loop_idx = df["圈数"]
    traj_id = df["轨迹ID"]
    v_x = df["v_x"]
    v_y = df["v_y"]
    v_z = df["v_z"]

    # 封装并返回
    return np.vstack((t, r, theta, phi, v_r, loop_idx, traj_id, v_x, v_y, v_z)).T


# numpy array 到 panda data_frame 转换函数
def np_to_pd(np_array, mode="RECT"):

    t = np_array[:, 0]
    v_r = np_array[:, 4]
    loop_idx = np_array[:, 5]
    traj_id = np_array[:, 6]
    v_x = np_array[:, 7]
    v_y = np_array[:, 8]
    v_z = np_array[:, 9]

    if mode == "RECT":
        x = np_array[:, 1]
        y = np_array[:, 2]
        z = np_array[:, 3]
        # 封装并返回
        return pd.DataFrame(
            {
                "时间(s)": t,
                "x(m)": x,
                "y(m)": y,
                "z(m)": z,
                "径向速度（m/s）": v_r,
                "圈数": loop_idx,
                "轨迹ID": traj_id,
                "v_x": v_x,
                "v_y": v_y,
                "v_z": v_z,
            }
        )
    else:
        r = np_array[:, 1]
        theta = np.rad2deg(np_array[:, 2])
        phi = np.rad2deg(np_array[:, 3])
        return pd.DataFrame(
            {
                "时间(s)": t,
                "斜距(m)": r,
                "方位角（°）": theta,
                "俯仰角（°）": phi,
                "径向速度（m/s）": v_r,
                "圈数": loop_idx,
                "轨迹ID": traj_id,
                "v_x": v_x,
                "v_y": v_y,
                "v_z": v_z,
            }
        )


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
def predict(states_rect):
    global transition_matrix
    predicts_rect = np.copy(states_rect).T
    predicts_rect[1:4] = (
        transition_matrix @ np.vstack([predicts_rect[1:4], predicts_rect[7:10]])
    )[:3]
    return predicts_rect.T


# TODO match 和 kalman 之间可以设计一个更好的结构，用来传递匹配的结果
def match(currents, predicts, threshold=(1, 20)):
    global current_trajectory_id
    for i in range(currents.shape[0]):
        # 将当前的每一个点与上一圈的普通运动学预测相比较
        distances = np.linalg.norm(currents[i, 1:4] - predicts[:, 1:4], axis=1)
        if (
            threshold[0] < np.min(distances) < threshold[1]
            ## （特殊处理)
            and np.linalg.norm(currents[i, 1:4]) > 200 # 斜距 200m 以上的点才匹配 ##############################
            and currents[i, 3] > 0 # z轴大于 150m 以上才匹配
        ):
            # 如果匹配到了，则更新当前点的轨迹ID
            temp_idx = np.argmin(distances)
            currents[i, 6] = predicts[temp_idx, 6]
            # 更新直角坐标系速度
            currents[i, 7:10] = (currents[i, 1:4] - predicts[temp_idx, 1:4]) / (
                currents[i, 0] - predicts[temp_idx, 0]
            )
        else:
            # 没有匹配到，说明当前的点是新出现的，添加新的ID
            currents[i, 6] = current_trajectory_id
            current_trajectory_id += 1
    # print("---- after match ----")
    # print(currents)
    return currents


# 选取普通运动学预测中的邻近作为观测，更新kalman状态
# TODO 当前版本的多目标更新采用相同的协方差矩阵，后续版本应当考虑每一个目标都维护一个协方差矩阵
# version 1
def kalman_update(kf, obs_i, pred_i, previous_states, state_covariance=1.0 * np.eye(6)):

    # 重新组织两个数组，用于存储匹配上的观测和预测
    obs = []
    pred = []
    for i in range(obs_i.shape[0]):
        if not np.isnan(obs_i[i, 7]):
            # 如果线速度不为nan，说明有匹配
            obs.append(obs_i[i])
            # 通过轨迹ID来查找对应的pred
            pred.append(
                previous_states[np.where(previous_states[:, 6] == obs_i[i, 6])[0][0]]
            )
    obs = np.array(obs).T
    pred = np.array(pred).T
    # print(obs, pred)
    print(obs.shape, pred.shape)
    print(np.vstack((pred[1:4, :], pred[7:10, :])).shape)
    print(np.vstack((obs[1:4, :], obs[7:10, :])).shape)
    # print(np.vstack((pred[1:4, :], pred[7:10, :])))
    # TODO recheck this part. This function may have the ability to handle multi data
    states_mean = np.zeros((6, obs.shape[1]))
    for i in range(obs.shape[1]):
        print(np.hstack([pred[1:4, i], pred[7:10, i]]))
        print(np.hstack([obs[1:4, i], obs[7:10, i]]))
        states_mean[:, i], state_covariance = kf.filter_update(
            filtered_state_mean=np.hstack([pred[1:4, i], pred[7:10, i]]),
            filtered_state_covariance=state_covariance,
            observation=np.hstack([obs[1:4, i]]),
        )

    return np.array([obs[0], states_mean, obs[7:10]]), state_covariance


def kalman_update_2(
    kf, obs, current_loop_data_polar, previous_states, state_covariance=1.0 * np.eye(6)
):

    for i in range(obs.shape[0]):
        if not np.isnan(obs[i, 7]):
            # 如果线速度不为nan，说明有匹配
            # 通过轨迹ID来查找对应的pred
            pred = previous_states[np.where(previous_states[:, 6] == obs[i, 6])[0][0]]
            temp_obs, state_covariance = kf.filter_update(
                filtered_state_mean=np.hstack([pred[1:4], pred[7:10]]),
                filtered_state_covariance=state_covariance,
                observation=obs[i, 1:4],
            )
            obs[i, 1:4] = temp_obs[:3]
        else:
            # 如果没有匹配上，说明是新出现的点，直接更新径向速度，状态不变
            obs[i] = update_vel(current_loop_data_polar[i], obs[i])

    return obs, state_covariance


def update_vel(*args):
    if len(args) == 1:
        states_polar = args[0]
        # 提取重复计算的部分
        v_r = states_polar.T[4].ravel()
        cos_theta = np.cos(states_polar.T[2])
        sin_theta = np.sin(states_polar.T[2])
        cos_phi = np.cos(states_polar.T[3])
        sin_phi = np.sin(states_polar.T[3])
        v_x = v_r * cos_theta * cos_phi
        v_y = v_r * sin_theta * cos_phi
        v_z = v_r * sin_phi

        return v_x, v_y, v_z
    elif len(args) == 2:
        states_polar = args[0]
        states_rect = args[1]

        v_r = states_polar[4]
        cos_theta = np.cos(states_polar[2])
        sin_theta = np.sin(states_polar[2])
        cos_phi = np.cos(states_polar[3])
        sin_phi = np.sin(states_polar[3])
        v_x = v_r * cos_theta * cos_phi
        v_y = v_r * sin_theta * cos_phi
        v_z = v_r * sin_phi
        states_rect[7:10] = np.array([v_x, v_y, v_z])

        return states_rect
    else:
        print("BAD update_vel inputs!/nDo nothing")
        return args


## TODO 如果点实在是太少的话，可以考虑把 trajectory ID 设置为 -1，表示噪声
def process(data, threshold=(1, 20)):
    global first_loop_index
    global total_loops
    global transition_matrix
    global current_trajectory_id

    # 初始化滤波器
    kf = initialize_kalman_filter()

    # 初始化状态协方差矩阵
    state_covariance = 1.0 * np.eye(6)

    # 初始帧处理，获取初始帧预测，预测根据为径向速度
    first_loop_data = data[data["圈数"] == first_loop_index] # 
    # 增加轨迹ID列，用于存储轨迹ID
    first_loop_data.loc[:, "轨迹ID"] = range(
        current_trajectory_id, current_trajectory_id + len(first_loop_data)
    )
    current_trajectory_id += len(first_loop_data)

    # 得到第一圈预测用的速度
    # 转换数据格式到np，转换坐标系到直角坐标系，以方便后续处理
    #  0  1    2     3    4     5     6        7nan 8nan 9nan
    # (t, r, theta, phi, v_r, lp_idx, traj_id, v_x, v_y, v_z)
    first_loop_states_polar = pd_to_np(first_loop_data)
    #  0  1  2  3   4     5     6        7nan 8nan 9nan
    # (t, x, y, z, v_r, lp_idx, traj_id, v_x, v_y, v_z)
    previous_states = polar_to_rect(first_loop_states_polar)

    # 更新预测中的速度
    v_x, v_y, v_z = update_vel(first_loop_states_polar)
    #  0  1  2  3   4     5     6         7    8    9
    # (t, x, y, z, v_r, lp_idx, traj_id, v_x, v_y, v_z)
    previous_states[:, 7:10] = np.array([v_x, v_y, v_z]).T

    # 累计记录数据
    accumulate = [np_to_pd(rect_to_polar(previous_states), mode="POLAR")]

    # 执行普通运动学预测，这里的previous_predicts是相对于下一帧（当前帧）来说的
    previous_predicts = predict(previous_states)

    for i in range(first_loop_index, total_loops  + first_loop_index):
        # (t, r, theta, phi, v_r, lp_idx, traj_id, v_x, v_y, v_z))
        current_loop_data_polar = pd_to_np(data[data["圈数"] == i])
        # (t, x, y, z, v_r, lp_idx, traj_id, v_x, v_y, v_z))
        current_loop_data_rect = polar_to_rect(current_loop_data_polar)
        # print(current_loop_data_rect)
        # 匹配当前帧所有点与上一帧预测，这里会加入匹配到的点的速度
        current_after_match = match(
            current_loop_data_rect, previous_predicts, threshold
        )
        # 通过匹配结果执行kalman滤波器更新, 匹配的结果作为卡尔曼的观测
        # 同时，在这个版本的卡尔曼中，一同完成了非匹配点的速度更新
        current_after_kalman, state_covariance = kalman_update_2(
            kf,
            current_after_match,
            current_loop_data_polar,
            previous_states,
            state_covariance,
        )

        # 加入累计数组
        accumulate.append(np_to_pd(rect_to_polar(current_after_kalman), mode="POLAR"))

        # 更新上一帧的状态
        previous_states = np.copy(current_after_kalman)

        # 继续执行普通运动学预测，为下一圈做好准备
        previous_predicts = predict(previous_states)

    return accumulate

# 使用函数修饰符可能会减少重构代价
# 时间记录封装
def time_record(func):
    start_time = time.time()  # 记录开始时间
    
    
    
    end_time = time.time()  # 记录结束时间
    execution_time = end_time - start_time  # 计算运行时间
    print("运行时间：", execution_time, " s")


def pre_process():
    pass

# TODO 增加轨迹权重，当轨迹越长，则减小最小阈值，增加最大阈值
# TODO 可以考虑增加速度滤波，大部分噪声的速度都很小，可以通过速度的大小来过滤噪声
# TODO 可以考虑容忍一两帧的阈值外，否则这个强阈值要求太难了
# TODO 考虑当前点的预测点搜索只在邻近的斜距、方位角、俯仰角范围内进行，这样可以减少计算量
if __name__ == "__main__":

    start_time = time.time()  # 记录开始时间
    # 加载文件数据
    input_file_path = (
        f"../materials/{file_name}"
    )
    output_file_path = (
        f"./output/finals/results_{file_name}_{threshold_top[0]}_{threshold_top[1]}_{threshold_top[2]}.xlsx"
    )
    # input_file_path = "../materials/赛道一：小、微无人机集群目标跟踪/点迹数据2-公开提供.xlsx"
    # output_file_path = f"./output/kalman_results_dataset2_{threshold[0]}_{threshold[1]}.xlsx"
    # input_file_path = "../materials/赛道一：小、微无人机集群目标跟踪/点迹数据3-公开提供.xlsx"
    # output_file_path = f"./output/kalman_results_dataset3_{threshold[0]}_{threshold[1]}.xlsx"

    print("输入文件：", input_file_path)
    print("阈值：", threshold_top)
    origin_data = pd.read_excel(input_file_path)

    # 选取需要处理的帧
    # loops_data = origin[origin["圈数"] <= total_loops].reset_index(drop=True)
    selected_data = origin_data[origin_data["圈数"] <= total_loops]

    # 添加新列
    selected_data.loc[:, "轨迹ID"] = np.nan
    selected_data.loc[:, "v_x"] = np.nan
    selected_data.loc[:, "v_y"] = np.nan
    selected_data.loc[:, "v_z"] = np.nan
    end_time = time.time()  # 记录结束时间
    pre_exe_time = end_time - start_time  # 计算运行时间
    print("预处理时间：", pre_exe_time, " s")

    start_time = time.time()  # 记录开始时间
    # 处理函数入口
    accumulate = process(selected_data, threshold_top)
    end_time = time.time()  # 记录结束时间
    execution_time = end_time - start_time  # 计算运行时间
    print("运行时间：", execution_time, " s")

    # 存储为 Excel 文件
    start_time = time.time()  # 记录开始时间
    combined = pd.concat(accumulate, ignore_index=True)
    combined.to_excel(output_file_path, index=False)
    end_time = time.time()  # 记录结束时间
    post_exe_time = end_time - start_time  # 计算运行时间
    print("后处理时间：", post_exe_time, " s")

    print("完成！")

