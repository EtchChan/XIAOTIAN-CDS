###
# python=3.8.5
# version=0.0.1
# workspace_root=$(project_root)
# This shit version is also in a MASS!
###

import pandas as pd
import numpy as np
from pykalman import KalmanFilter

# Function to convert from polar to Cartesian coordinates
def polar_to_cartesian(r, theta, phi):
    # Convert polar coordinates (r, theta, phi) to Cartesian coordinates (x, y, z)
    x = r * np.cos(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.cos(phi)
    z = r * np.sin(phi)
    return x, y, z

# Function to convert from Cartesian to polar coordinates
def cartesian_to_polar(x, y, z):
    # Convert Cartesian coordinates (x, y, z) to polar coordinates (r, theta, phi)
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(y, x)
    phi = np.arctan2(z, np.sqrt(x**2 + y**2))
    return r, theta, phi

def initialize_kalman_filter():
    # 初始化卡尔曼滤波器，状态包括位置和速度
    # (r, theta, phi, v_r, w_theta, w_phi)
    # 假设 w_theta, w_phi 这两项为 0
    transition_matrix = np.array(
        [
            [1, 0, 0, 3, 0, 0],  # 3s 一圈
            [0, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 1],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ]
    )

    observation_matrix = np.eye(3, 6)

    kf = KalmanFilter(
        transition_matrices=transition_matrix,
        observation_matrices=observation_matrix,
        initial_state_mean=np.zeros(6),
        initial_state_covariance=1.0 * np.eye(6),
        transition_covariance=0.1 * np.eye(6),
        observation_covariance=0.1 * np.eye(3),
    )

    return kf


def predict_next_state(kf, point):
    # 初始状态包括位置和速度
    state_mean = np.hstack((point[:4], 0, 0))  # 最后两个方向速度为 0
    state_covariance = 1.0 * np.eye(6)

    state_mean, state_covariance = kf.filter_update(
        filtered_state_mean=state_mean,
        filtered_state_covariance=state_covariance,
        observation=point[:3],
    )
    return state_mean[:3]


def match_points(predicted_points, next_frame_points, threshold=2.5):
    loop_index = next_frame_points["圈数"].iloc[0]
    # extract the data
    next_r = next_frame_points["斜距(m)"]
    next_theta = np.deg2rad(next_frame_points["方位角（°）"])
    next_phi = np.deg2rad(next_frame_points["俯仰角（°）"])
    next_v_r = next_frame_points["径向速度（m/s）"]

    # Calculate Cartesian coordinates
    next_x = next_r * np.cos(next_theta) * np.cos(next_phi)
    next_y = next_r * np.sin(next_theta) * np.cos(next_phi)
    next_z = next_r * np.sin(next_phi)

    next_state = np.vstack((next_x, next_y, next_z)).T

    pred_r = predicted_points.T[0]
    pred_theta = np.deg2rad(predicted_points.T[1])
    pred_phi = np.deg2rad(predicted_points.T[2])

    pred_x = pred_r * np.cos(pred_theta) * np.cos(pred_phi)
    pred_y = pred_r * np.sin(pred_theta) * np.cos(pred_phi)
    pred_z = pred_r * np.sin(pred_phi)

    pred_state = np.vstack((pred_x, pred_y, pred_z)).T

    print("---- match ----")
    print("---- pred state (x, y, z)")
    print(pred_state)
    print("---- next state (x, y, z)")
    print(next_state)

    # 匹配点
    matched_points = []
    for pred in pred_state:
        distances = np.linalg.norm(next_state - pred, axis=1)
        if np.min(distances) < threshold:
            # 补充上匹配点下一帧的速度，用于后面的预测任务
            temp_index = next_v_r.index[0] + np.argmin(distances)
            matched_points.append(
                # np.append(pred, next_v_r[temp_index]) # 保留预测
                np.array([next_x[temp_index], next_y[temp_index], next_z[temp_index], next_v_r[temp_index]]) # 保留观测
            )
    print("number of matched points (x, y, z):", len(matched_points))
    print(matched_points)

    # convert back to (r, theta, phi)
    temp_index = 0
    for x, y, z, v in matched_points:
        point_r = np.sqrt(x**2 + y**2 + z**2)
        point_theta = np.rad2deg(np.arctan2(y, x))
        point_phi = np.rad2deg(np.arcsin(z / point_r))
        matched_points[temp_index] = np.array(
            [point_r, point_theta, point_phi, v, loop_index]
        )
        temp_index += 1

    return pd.DataFrame(
        matched_points,
        columns=["斜距(m)", "方位角（°）", "俯仰角（°）", "径向速度（m/s）", "圈数"],
    )


#
def process_point_clouds(kf, loop_data):
    # output is a numpy array with dimension nx3
    # only contain the coordinate

    # extract the data
    r = loop_data["斜距(m)"]
    theta = loop_data["方位角（°）"]
    phi = loop_data["俯仰角（°）"]
    v_r = loop_data["径向速度（m/s）"]

    # Now the data in state is listed as:
    # r theta phi v_r
    # r theta phi v_r
    # . .     .   .
    # . .     .   .
    state = np.vstack((r, theta, phi, v_r)).T

    predicted_state = np.zeros([r.shape[0], 3])

    for i in range(0, r.shape[0]):
        predicted_state[i] = predict_next_state(kf, state[i])

    print("----predicted state (r, theta, phi):")
    print(predicted_state)

    return predicted_state

def assign_trajectory_ids(radar_data, distance_threshold=2.5):
    # Initialize the first trajectory ID
    current_trajectory_id = 0
    
    # Initialize a list to store the trajectory IDs
    radar_data['轨迹ID'] = np.nan
    
    # Loop through the radar data by scan cycles
    for loop_num in sorted(radar_data['圈数'].unique()):
        current_frame = radar_data[radar_data['圈数'] == loop_num]
        
        if loop_num == min(radar_data['圈数']):  # First loop, assign new IDs
            radar_data.loc[radar_data['圈数'] == loop_num, '轨迹ID'] = range(current_trajectory_id, current_trajectory_id + len(current_frame))
            current_trajectory_id += len(current_frame)
        else:
            previous_frame = radar_data[radar_data['圈数'] == (loop_num - 1)]
            
            for idx, point in current_frame.iterrows():
                # Convert both frames' points to Cartesian coordinates
                prev_x, prev_y, prev_z = polar_to_cartesian(previous_frame['斜距(m)'], np.deg2rad(previous_frame['方位角（°）']), np.deg2rad(previous_frame['俯仰角（°）']))
                curr_x, curr_y, curr_z = polar_to_cartesian(point['斜距(m)'], np.deg2rad(point['方位角（°）']), np.deg2rad(point['俯仰角（°）']))
                
                # Compute distances between current point and all points in the previous frame
                distances = np.sqrt((prev_x - curr_x)**2 + (prev_y - curr_y)**2 + (prev_z - curr_z)**2)
                
                # Check if there is a match within the threshold
                if distances.min() < distance_threshold:
                    matched_index = distances.argmin()
                    radar_data.loc[idx, '轨迹ID'] = radar_data.loc[previous_frame.index[matched_index], '轨迹ID']
                else:
                    # Assign a new trajectory ID
                    radar_data.loc[idx, '轨迹ID'] = current_trajectory_id
                    current_trajectory_id += 1

    return radar_data


def process_multiple_frames(loops_data):

    # 仅在此初始化滤波器
    kf = initialize_kalman_filter()

    # 初始帧处理，获取初始帧预测，预测来源与径向速度
    current_loop_data = loops_data[loops_data["圈数"] == 1]
    current_predictions = process_point_clouds(kf, current_loop_data)

    # combined matched points 存储了从第一帧开始所有的匹配点（第一帧则为所有点）
    combined_matched_points = [current_loop_data]
    # 从第2帧开始循环遍历各帧
    for i in range(2, total_loops + 1):
        print(f"############# current frame: {i} ###############")
        next_loop_data = loops_data[loops_data["圈数"] == i]
        matched_points = match_points(current_predictions, next_loop_data, 100)
        print(f"Matched point for frame {i}:\n ", matched_points)
        combined_matched_points.append(matched_points)
        current_predictions = process_point_clouds(kf, matched_points)

    # 存储为 Excel 文件
    combined = pd.concat(combined_matched_points, ignore_index=True)
    combined.to_excel("output.xlsx", index=False)

    return current_predictions


if __name__ == "__main__":

    # Load the Excel file
    file_path = "../materials/赛道一：小、微无人机集群目标跟踪/点迹数据1-公开提供.xlsx"  # Update this to the correct file path
    df = pd.read_excel(file_path)

    # total processed loops
    total_loops = 20

    # 选取需要处理的帧
    loops_data = df[df["圈数"] <= total_loops].reset_index(drop=True)

    # 多帧处理
    # matched_points = process_multiple_frames(loops_data)
    
    # 轨迹ID
    # processed_data = assign_trajectory_ids(loops_data, distance_threshold=100)
    # processed_data.to_excel("output2.xlsx", index=False)
