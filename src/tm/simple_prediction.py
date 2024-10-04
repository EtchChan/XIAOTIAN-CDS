###
# python=3.8.5
# version=0.0.1
# workspace_root=$(project_root)
# Do NOT run!!! This file is not runnable!!!
###

import pandas as pd
import numpy as np

def predict_next_state(kf, point):
    # 初始状态包括位置和速度
    state_mean = np.hstack((point[:3], [point[3]], 0, 0))  # 最后两个方向速度为 0
    state_covariance = 1.0 * np.eye(6)
    
def match_points(predicted_points, next_frame_points, threshold=0.5):
    # 匹配点
    matched_points = []
    for pred in predicted_points:
        distances = np.linalg.norm(next_frame_points - pred, axis=1)
        if np.min(distances) < threshold:
            matched_points.append(pred)
    return matched_points

# 
def process_point_clouds(kf, loop_data):
    
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
    
    for i in range (0, r.shape[0]):
        predicted_state[i] = predict_next_state(kf, state[i])
    
    print("predicted state:")
    print(predicted_state)
    
    return predicted_state

def process_multiple_frames(loops_data):
    kf = initialize_kalman_filter()

    # 初始帧处理
    current_loop_data = loops_data[loops_data["圈数"] == 1]
    
    # 初始帧处理
    current_predictions = process_point_clouds(kf, current_loop_data)
    
    # 循环遍历各帧
    for i in range (2, total_loops + 1):
        next_loop_data = loops_data[loops_data["圈数"] == i]
        matched_points = match_points(current_loop_data, next_loop_data)
        print(f"Matched point for frame {i}: ", matched_points)
        
        current_predictions = process_point_clouds(kf, matched_points)
    
    return current_predictions


if __name__ == "__main__":
    
    
    # Load the Excel file
    file_path = "../materials/赛道一：小、微无人机集群目标跟踪/点迹数据1-公开提供.xlsx"  # Update this to the correct file path
    df = pd.read_excel(file_path)

    # total processed loops
    total_loops = 5
    
    # filtering
    loops_data = df[df["圈数"] <= total_loops].reset_index(drop=True)

    # 多帧处理
    matched_points = process_multiple_frames(loops_data)



    ################ 
    # obtained from the poe
    # 示例多帧点云数据（三维坐标 + 径向速度）
    # frames = [
    #     np.array([[1.0, 0.5, 0.3, 0.1], [2.0, 1.0, 0.4, 0.2], [3.0, 1.5, 0.5, 0.3]]),
    #     np.array([[1.1, 0.5, 0.3, 0.1], [2.1, 1.0, 0.4, 0.2], [3.1, 1.5, 0.5, 0.3]]),
    #     np.array([[1.2, 0.5, 0.3, 0.1], [2.2, 1.0, 0.4, 0.2], [3.2, 1.5, 0.5, 0.3]])
    # ]

    # process_multiple_frames(frames)