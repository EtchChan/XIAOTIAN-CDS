###
# python=3.8.5
# version=0.0.1
# workspace_root=$(project_root)
# Simple test. Do not run!! This program is no longer under maintain.
###

import numpy as np
from pykalman import KalmanFilter

def initialize_kalman_filter():
    # 初始化卡尔曼滤波器，状态包括位置和速度
    transition_matrix = np.array([
        [1, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1]
    ])
    
    observation_matrix = np.eye(3, 6)
    
    kf = KalmanFilter(
        transition_matrices=transition_matrix,
        observation_matrices=observation_matrix,
        initial_state_mean=np.zeros(6),
        initial_state_covariance=1.0 * np.eye(6),
        transition_covariance=0.1 * np.eye(6),
        observation_covariance=0.1 * np.eye(3)
    )
    
    return kf

def predict_next_state(kf, point):
    # 初始状态包括位置和速度
    state_mean = np.hstack((point[:3], [point[3]] * 3))  # 假设速度在各方向相同
    state_covariance = 1.0 * np.eye(6)
    
    state_mean, state_covariance = kf.filter_update(
        filtered_state_mean=state_mean,
        filtered_state_covariance=state_covariance,
        observation=point[:3]
    )
    return state_mean[:3]

def match_points(predicted_points, next_frame_points, threshold=0.5):
    # 匹配点
    matched_points = []
    for pred in predicted_points:
        distances = np.linalg.norm(next_frame_points - pred, axis=1)
        if np.min(distances) < threshold:
            matched_points.append(pred)
    return matched_points

def process_point_clouds(frame1_points, frame2_points):
    kf = initialize_kalman_filter()
    
    # 对第一帧每个点进行预测
    predicted_points = np.array([predict_next_state(kf, point) for point in frame1_points])
    
    # 匹配预测点与第二帧点云
    matched_points = match_points(predicted_points, frame2_points)
    
    return matched_points


if __name__ == "__main__":
    # 示例点云数据（三维坐标 + 径向速度）
    frame1_points = np.array([[1.0, 0.5, 0.3, 0.1], [2.0, 1.0, 0.4, 0.2], [3.0, 1.5, 0.5, 0.3]])
    frame2_points = np.array([[1.1, 0.5, 0.3], [2.1, 1.0, 0.4], [3.1, 1.5, 0.5]])

    matched_points = process_point_clouds(frame1_points, frame2_points)
    print("Matched Points:", matched_points)