###
# python=3.8.5
# version=0.1.0
# workspace_root=$(project_root)
# This shit version is in a MASS!
###


import pandas as pd
import numpy as np
import plotly.graph_objs as go

from utilities import read_config, polar_to_rect, rect_to_polar, pd_to_np, np_to_pd
from utilities import colors


def plot_graph(df, config):

    start_loop = config["start_loop"]
    if config["end_loop"] == -1:
        end_loop = np.max(df["loop"])
    else:
        end_loop = config["end_loop"]
    loops_data = df[(df["loop"] >= start_loop) & (df["loop"] <= end_loop)]
    # Get xyz data
    loops_data = np_to_pd(polar_to_rect(pd_to_np(loops_data)))
    # Create traces
    traces = []

    if config["data_type"] in ["raw", "ransac_processed", "centers"]:
        # 如果是原始数据，不进行轨迹表示
        for loop_index in range(start_loop, end_loop + 1):
            loop_data = loops_data[loops_data["loop"] == loop_index]
            marker_dict = dict(
                size=5,
                color=loop_data["loop"],  # 使用 loop 数据作为颜色映射
                colorscale="Cividis",  # 颜色渐变
                cmin=start_loop,
                cmax=end_loop,
                # colorbar=dict(title="Loop Index"),
            )  # Other options: Cividis, Plasma, Inferno, Viridis
            trace = go.Scatter3d(
                x=loop_data["x"],
                y=loop_data["y"],
                z=loop_data["z"],
                mode="markers",
                marker=marker_dict,
                name=f"Loop {loop_index}",
                text=[
                    f"Loop: {loop_index}<br>Range: {dist}m<br>Azimuth: {azim}°<br>Pitch: {pitch}°<br>({x_i, y_i, z_i})"
                    for dist, azim, pitch, x_i, y_i, z_i in zip(
                        loop_data["r"],
                        loop_data["azimuth"],
                        loop_data["pitch"],
                        loop_data["x"],
                        loop_data["y"],
                        loop_data["z"],
                    )
                ],
                hoverinfo="text",
            )

            traces.append(trace)
    # elif config["data_type"] == "ransac_processed":
    #     # This is deprecated
    #     # See ransac.py for details
    #     print(
    #         "[WARNING] ransac.py is deprecated in this project!\n\
    #             Refer to ransac.py for more details"
    #     )
    #     pass
    elif config["data_type"] == "kalman_processed":
        # 否则用轨迹
        # TODO 之后需要按照新生成数据格式来更改这一部分
        # 找到轨迹列中出现次数最多的10个轨迹
        top_10_trajectories = df["id"].value_counts().head(15)
        print("Top 15 trajectories: ", top_10_trajectories)

        # 转成numpy
        top_10_trajectories = np.array(
            [top_10_trajectories.index, top_10_trajectories.values]
        ).T

        # total_loops = 10  # for there is only 5 colors, this variable should be <=5
        # Loop over the first ${total_loops} and create a separate trace for each
        for id in range(15):
            loop_data = loops_data[loops_data["id"] == top_10_trajectories[id][0]]

            loop_index = loop_data["loop"]
            # id = loop_data["轨迹ID"]

            # Create a 3D scatter plot trace for this loop
            marker_dict = dict(size=5, color=colors[id], opacity=0.8)
            trace = go.Scatter3d(
                x=loop_data["x"],
                y=loop_data["y"],
                z=loop_data["z"],
                mode="markers",
                marker=marker_dict,
                name=f"id: {top_10_trajectories[id][0]}",
                text=[
                    f"Loop: {loop_index}<br>Range: {dist}m<br>Azimuth: {azim}°<br>Pitch: {pitch}°<br>({x_i, y_i, z_i})"
                    for loop_index, dist, azim, pitch, x_i, y_i, z_i in zip(
                        loop_index,
                        loop_data["r"],
                        loop_data["azimuth"],
                        loop_data["pitch"],
                        loop_data["x"],
                        loop_data["y"],
                        loop_data["z"],
                    )
                ],
                hoverinfo="text",
            )

            traces.append(trace)
    else:
        print(
            "[WARNING] Received wrong data type. Please check your configuration file"
        )

    # 尝试ransac失败
    # ransac data1
    # 可视化参数方程形式的直线
    # x0 = 1317.6652994869933
    # y0 = -184.12395974969374
    # z0 = 163.40344596709195
    # dx = 1
    # dy = -0.09853806830438212
    # dz = 0.04107776043869889
    # t = np.linspace(-1300, 1700, 3000)
    # x_line = x0 + t * dx
    # y_line = y0 + t * dy
    # z_line = z0 + t * dz

    # line_trace = go.Scatter3d(
    #     x=x_line,
    #     y=y_line,
    #     z=z_line,
    #     mode="lines",
    #     line=dict(color="red", width=4),
    #     name="参数方程直线",
    # )
    # traces.append(line_trace)

    if config["data_type"] == "centers":
        file_path = f"output/finals/{config['file_name']}_{config['start_loop']}_{config['end_loop']}_{config['threshold']}_center_trajectory.xlsx"
        df = pd.read_excel(file_path)
        # 创建 3D 散点图
        trace = go.Scatter3d(
            x=df["群中心X坐标(m)"],
            y=df["群中心Y坐标(m)"],
            z=df["群中心Z坐标(m)"],
            mode="markers",
            marker=dict(
                size=5,
                color="red",
            ),
            text=[
                f"时间: {time}s<br>斜距: {dist}m<br>方位角: {azim}°<br>高低角: {pitch}°<br>群规模: {size}<br>圈数: {loop}"
                for time, dist, azim, pitch, size, loop in zip(
                    df["时间(s)"],
                    df["群中心斜距(m)"],
                    df["群中心方位角(°)"],
                    df["群中心高低角(°)"],
                    df["群规模"],
                    df["圈数"],
                )
            ],
            hoverinfo="text",
        )
        traces.append(trace)

    # Reference azimuth line (horizontal)
    azimuth_line = go.Scatter3d(
        x=[np.min(loops_data["x"]), np.max(loops_data["x"])],
        y=[0, 0],
        z=[0, 0],
        mode="lines",
        line=dict(color="blue", width=4),
        name="Azimuth Line",
    )

    # Reference pitch line (horizontal)
    azimuth_line_2 = go.Scatter3d(
        x=[0, 0],
        y=[np.min(loops_data["y"]), np.max(loops_data["y"])],
        z=[0, 0],
        mode="lines",
        line=dict(color="green", width=4),
        name="azimuth Line 2",
    )

    # Reference pitch line (vertical)
    pitch_line = go.Scatter3d(
        x=[0, 0],
        y=[0, 0],
        z=[np.min(loops_data["z"]), np.max(loops_data["z"])],
        mode="lines",
        line=dict(color="red", width=4),
        name="Pitch Line",
    )

    # Add reference lines to traces
    traces.extend([azimuth_line, azimuth_line_2, pitch_line])

    # Create the layout for the 3D plot
    layout = go.Layout(
        title=dict(
            text=f"{config['file_name']}_{config['threshold']}",
            font=dict(size=50),
        ),
        scene=dict(
            xaxis=dict(title="X (meters)"),
            yaxis=dict(title="Y (meters)"),
            zaxis=dict(title="Z (meters)"),
            # data_3
            # xaxis=dict(title="X (meters)", range=[0, 4000]),
            # yaxis=dict(title="Y (meters)", range=[-1000, 1000]),
            # zaxis=dict(title="Z (meters)", range=[0, 400]),
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

    # 读取配置文件
    # config = read_config("src/tm/config/final_visual_data_1.yaml")
    # config = read_config("src/tm/config/final_visual_data_1_ransac.yaml")
    # config = read_config("src/tm/config/final_visual_data_3_filtered.yaml")
    config = read_config("src/tm/config/final_visual_data_3_centers.yaml")
    print("Visualization configuration:")
    print(config)

    file_path = None
    # 读取数据文件
    if config["data_type"] in ["raw", "ransac_processed"]:
        file_path = f"../materials/finals/{config['file_name']}.xlsx"  # Update this to the correct file path
    elif config["data_type"] in ["kalman_processed", "centers"]:
        file_path = f"output/finals/{config['file_name']}_{config['start_loop']}_{config['end_loop']}_{config['threshold']}.xlsx"  # Update this to the correct file path

    df = pd.read_excel(file_path)
    # 新的数据已经统一了列标准，因此这里可以不用更改
    # 列标准
    # df.columns = [
    #     "time",
    #     "r",
    #     "azimuth",
    #     "pitch",
    #     "v_r",
    #     "loop",
    #     "id",
    #     "x",
    #     "y",
    #     "z",
    #     "v_x",
    #     "v_y",
    #     "v_z",
    # ]

    # 此函数只画图
    plot_graph(df, config)
