###
# python=3.8.5
# version=0.0.2
# workspace_root=$(project_root)
# This shit version is in a MASS!
###

import pandas as pd
import numpy as np
import plotly.graph_objs as go


def plot_graph(df, plot_title):

    # Filter the data for only the first 5 loops (圈数 <= 5)
    # 圈数超过表格中最大圈数则代表所有圈数均进行处理
    loops_data = df[df["圈数"] <= 400].reset_index(drop=True)

    # Convert spherical coordinates to Cartesian coordinates
    r = loops_data["斜距(m)"]
    theta = np.deg2rad(loops_data["方位角（°）"])  # Convert degrees to radians
    phi = np.deg2rad(loops_data["俯仰角（°）"])  # Convert degrees to radians

    # 找到轨迹列中出现次数最多的10个轨迹
    top_10_trajectories = df["轨迹ID"].value_counts().head(10)
    # print("Top 10 trajectories: ", top_10_trajectories)

    # 转成numpy
    top_10_trajectories = np.array(
        [top_10_trajectories.index, top_10_trajectories.values]
    ).T

    # -----------------------
    # 这一部分好像并没有用

    # Calculate Cartesian coordinates
    x = r * np.cos(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.cos(phi)
    z = r * np.sin(phi)

    # Prepare the data for KNN clustering
    xyz_data = np.vstack((x, y, z)).T

    # -----------------------

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
        "#FFA500",  # Orange
    ]

    # Create traces for each cluster
    traces = []

    # total_loops = 10  # for there is only 5 colors, this variable should be <=5
    # Loop over the first ${total_loops} and create a separate trace for each
    for id in range(10):
        loop_data = loops_data[loops_data["轨迹ID"] == top_10_trajectories[id][0]]

        # Convert spherical coordinates to Cartesian coordinates
        r = loop_data["斜距(m)"]
        theta = np.deg2rad(loop_data["方位角（°）"])  # Convert degrees to radians
        phi = np.deg2rad(loop_data["俯仰角（°）"])  # Convert degrees to radians
        loop_index = loop_data["圈数"]
        # id = loop_data["轨迹ID"]

        # Calculate Cartesian coordinates
        x = r * np.cos(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.cos(phi)
        z = r * np.sin(phi)

        next_state = np.vstack((x, y, z)).T

        # Create a 3D scatter plot trace for this loop

        marker_dict = dict(size=5, color=colors[id], opacity=0.8)
        trace = go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="markers",
            marker=marker_dict,
            name=f"id: {top_10_trajectories[id][0]}",
            text=[
                f"Loop: {loop_number}<br>Range: {dist}m<br>Azimuth: {azim}°<br>Pitch: {pitch}°<br>({x_i, y_i, z_i})"
                for loop_number, dist, azim, pitch, x_i, y_i, z_i in zip(
                    loop_index,
                    r,
                    loop_data["方位角（°）"],
                    loop_data["俯仰角（°）"],
                    x,
                    y,
                    z,
                )
            ],
            hoverinfo="text",
        )

        traces.append(trace)

    # Reference azimuth line (horizontal)
    azimuth_line = go.Scatter3d(
        x=[0, np.max(x)],
        y=[0, 0],
        z=[0, 0],
        mode="lines",
        line=dict(color="blue", width=4),
        name="Azimuth Line",
    )

    # Reference pitch line (vertical)
    pitch_line = go.Scatter3d(
        x=[0, 0],
        y=[0, 0],
        z=[0, np.max(z)],
        mode="lines",
        line=dict(color="red", width=4),
        name="Pitch Line",
    )

    # Add reference lines to traces
    traces.extend([azimuth_line, pitch_line])

    # Create the layout for the 3D plot
    layout = go.Layout(
        title=dict(
            text=plot_title,
            font=dict(size=50),
        ),
        scene=dict(
            xaxis=dict(title="X (meters)"),
            yaxis=dict(title="Y (meters)"),
            zaxis=dict(title="Z (meters)"),
        ),
        showlegend=True,
        legend=dict(
            font=dict(size=40),
        ),
    )

    # Create the figure with all traces
    fig = go.Figure(data=traces, layout=layout)

    # Show the interactive plot
    fig.show()


if __name__ == "__main__":
    # Load the Excel file
    threshold = (6, 45, 200, 0)
    # file_path = "../materials/赛道一：小、微无人机集群目标跟踪/点迹数据1-公开提供.xlsx"  # Update this to the correct file path
    # file_path = "./output/output.xlsx"  # Update this to the correct file path
    # file_path = "./output/output2.xlsx"  # Update this to the correct file path
    # file_path = "./output/output3.xlsx"  # Update this to the correct file path
    # file_path = f"./output/kalman_results_dataset1_{threshold[0]}_{threshold[1]}.xlsx"
    # file_path = f"./output/kalman_results_dataset2_{threshold[0]}_{threshold[1]}.xlsx"
    # file_path = f"./output/kalman_results_dataset3_{threshold[0]}_{threshold[1]}.xlsx"
    
    
    file_path = f"./output/filtered/kalman_results_dataset2_{threshold[0]}_{threshold[1]}_{threshold[2]}_{threshold[3]}_filtered.xlsx"
    
    plot_title = "点迹数据-1，群规模为：1"
    print("输入文件：", file_path)
    df = pd.read_excel(file_path)

    plot_graph(df, plot_title)
