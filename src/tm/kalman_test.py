###
# python=3.8.5
# version=0.1.0
# workspace_root=$(project_root)
###

import itertools
import kalman_main_2
import visualization_2

import pandas as pd
import numpy as np
import time


if __name__ == "__main__":
    print("当前执行：kalman_test.py")
    start_time = time.time()  # 记录开始时间
    # 加载文件数据
    input_file_path = (
        "../materials/赛道一：小、微无人机集群目标跟踪/点迹数据3-公开提供.xlsx"
    )

    print("输入文件：", input_file_path)
    origin_data = pd.read_excel(input_file_path)

    # 选取需要处理的帧
    total_loops = 309
    # selected_data = origin_data[origin_data["圈数"] <= total_loops]
    # 1: 220 305
    selected_data = origin_data[
        ((origin_data["圈数"] >= 0) & (origin_data["圈数"] <= 305))
    ]
    # print(selected_data)
    # 添加新列
    selected_data.loc[:, "轨迹ID"] = np.nan
    selected_data.loc[:, "v_x"] = np.nan
    selected_data.loc[:, "v_y"] = np.nan
    selected_data.loc[:, "v_z"] = np.nan
    end_time = time.time()  # 记录结束时间
    pre_exe_time = end_time - start_time  # 计算运行时间
    print("预处理时间：", pre_exe_time, " s")

    # 网格
    # for i, j in itertools.product(range(5), range(30, 150, 15)):
    #     start_time = time.time()  # 记录开始时间
    #     output_file_path = f"./output/kalman_results_dataset1_{i}_{j}.xlsx"
    #     print(f"当前阈值：({i}, {j})")
    #     accumulate = kalman_main_2.process(selected_data, (i, j))
    #     end_time = time.time()  # 记录结束时间
    #     execution_time = end_time - start_time  # 计算运行时间
    #     print("运行时间：", execution_time, " s")

    #     # # 存储为 Excel 文件
    #     start_time = time.time()  # 记录开始时间
    #     combined = pd.concat(accumulate, ignore_index=True)
    #     combined.to_excel(output_file_path, index=False)
    #     end_time = time.time()  # 记录结束时间
    #     post_exe_time = end_time - start_time  # 计算运行时间
    #     print("后处理时间：", post_exe_time, " s")

    #     visualization_2.plot_graph(combined, f"点迹数据-1，群规模为：1，阈值：{i}，{j}")
    #     end_time = time.time()  # 记录结束时间
    #     execution_time = end_time - start_time  # 计算运行时间
    #     print("可视化时间：", execution_time, " s")

    # 1: 2 90, 4(5) 40 []; 5 30 [200];
    # 2: 4(6) 35 []; 4 35(40),  6 45(best), [200]
    # 3: 4-7(10) 80, 10 80(90), 7 95 [200]

    i = 10
    j = 80

    start_time = time.time()  # 记录开始时间
    output_file_path = f"./output/kalman_results_dataset3_{i}_{j}_200_0.xlsx"
    print(f"当前阈值：({i}, {j})")
    accumulate = kalman_main_2.process(selected_data, (i, j))
    end_time = time.time()  # 记录结束时间
    execution_time = end_time - start_time  # 计算运行时间
    print("运行时间：", execution_time, " s")

    # # 存储为 Excel 文件
    start_time = time.time()  # 记录开始时间
    combined = pd.concat(accumulate, ignore_index=True)
    combined.to_excel(output_file_path, index=False)
    end_time = time.time()  # 记录结束时间
    post_exe_time = end_time - start_time  # 计算运行时间
    print("后处理时间：", post_exe_time, " s")

    visualization_2.plot_graph(combined, f"点迹数据-3，群规模为：3，阈值：{i}，{j}")
    end_time = time.time()  # 记录结束时间
    execution_time = end_time - start_time  # 计算运行时间
    print("可视化时间：", execution_time, " s")

    print("完成！")
