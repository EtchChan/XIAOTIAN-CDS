###
# python=3.8.5
# version=0.0.2
# workspace_root=$(project_root)
# 
###


import pandas as pd

def extract_rows_by_trajectory_id(input_file, output_file, trajectory_ids):
    # 读取 Excel 文件
    df = pd.read_excel(input_file)

    # 过滤数据，只保留轨迹ID在指定列表中的行
    filtered_df = df[df['轨迹ID'].isin(trajectory_ids)]

    # 将过滤后的数据写入新的 Excel 文件
    filtered_df.to_excel(output_file, index=False)

    print(f"已将轨迹ID为 {trajectory_ids} 的行抽取到 {output_file}")

def main():
    
    # 指定参数
    threshold = (10, 80, 200, 0)
    input_file = f"./output/kalman_results_dataset3_{threshold[0]}_{threshold[1]}_{threshold[2]}_{threshold[3]}.xlsx"  # 输入的 Excel 文件路径
    output_file = f"./output/filtered/kalman_results_dataset3_{threshold[0]}_{threshold[1]}_{threshold[2]}_{threshold[3]}_filtered_3.xlsx"  # 输出的 Excel 文件路径
    # 5 30 200
    # dataset1: id0->14075, 15237, 16182
    #           id1->13170, 13878, 13538, 14000
    #           id2
    
    # 0 40 400 150
    # dataset1: t0: id0->2100, 4413
    #           t1: id1->1025, id2->640, id3->835, id4->426(partial)
    #           t2: id5->1946, id6->1873, id7->426(partial)
    #           t3: id8->2926
    #           t4: id9->4402
    # dataset2: t0: id0->1450, 1749, id1->1449, 1769, id2->1507, id3->1273, 1631
    # dataset3: t0: id0->3923, id1->5345
    #           t1: id2->18004(partial), id3->18841, 18004(partial)
    trajectory_ids = [18841]  # 需要抽取的轨迹ID列表

    extract_rows_by_trajectory_id(input_file, output_file, trajectory_ids)

if __name__ == "__main__":
    main()