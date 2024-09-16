"""
raw data processing functions set
Author: alfeak
Date: 2022-01-12
"""

import pandas as pd
import numpy as np
import os

def xlsx2csv(raw_data_path : str)-> None:
    """
    process raw data, transform xlsx file to csv file,
    save the file under raw_data directory

    Input:
    :param raw_data_path: raw data path
    :return: None
    """

    # check if raw data file exists
    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(f"raw data file {raw_data_path} not found")

    # get raw data file name and directory
    raw_data_file_name = os.path.basename(raw_data_path).split('.')[0]
    raw_data_dir = os.path.dirname(raw_data_path)

    # get processed data file path
    processed_data_path1 = os.path.join(raw_data_dir, f"{raw_data_file_name}.csv")

    # read raw data
    raw_data = pd.read_excel(raw_data_path,sheet_name=0,engine='openpyxl')
    cleaned_data = raw_data.drop(columns=['Unnamed: 7'], errors='ignore')
    raw_data.to_csv(processed_data_path1, 
                    index=False,
                    header=None,
                    sep='\t',
                    mode='w')

def csv2npy(raw_data_path : str)-> None:
    """
    process raw data, transform csv file to npy file,
    save the file under raw_data directory
    
    Input:
    :param raw_data_path: raw data path
    :return: None
    
    input data format:
    目标方位角(°)	目标斜距(m)	相对高度(m)	径向速率(m/s)	记录时间(s)	RCS	标签
    numpy data format:
    [N,B,7]
    N: number of track
    B: number of data in each track (2^n , last one for label)
    7: [记录时间,斜距,方位角,fuyangjiao,径向速率,quanshu,相对高度,RCS,标签] 
    """

    # check if raw data file exists
    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(f"raw data file {raw_data_path} not found")

    # get raw data file name and directory
    raw_data_file_name = os.path.basename(raw_data_path).split('.')[0]
    raw_data_dir = os.path.dirname(raw_data_path)

    # get processed data file path
    processed_data_path2 = os.path.join(raw_data_dir, f"{raw_data_file_name}.npy")

    # read raw data
    raw_data = pd.read_csv(raw_data_path, sep='\t', header=None, low_memory=False)
    raw_data = raw_data.values[:,:-1]
    data_seq_list = list()
    seq_num = 0
    for line in raw_data:
        if line[-1] != " ":
            if seq_num != 0:
                data_seq_list.append(seq_num)
            seq_num = 0
        seq_num += 1
    data_seq_list.append(seq_num)
    data_seq_list = np.array(data_seq_list)
    print("track total nums: " ,data_seq_list.shape[0])
    print("track min points nums: " , min(data_seq_list))
    print("track max points nums: " , max(data_seq_list))
    print("track mean points nums: " , np.mean(data_seq_list))

    #process data to npy format
    data_seq_length = 15
    data_track_list = list()
    data_seq_list = list()
    seq_num = 0
    last_label = -1
    for line in raw_data:
    #目标方位角(°)	目标斜距(m)	相对高度(m)	径向速率(m/s)	记录时间(s)	RCS	标签
    #[记录时间,斜距,方位角,fuyangjiao,径向速率,quanshu,相对高度,RCS,标签] 
        if line[-1] != " ":
            last_label = line[-1]
            if seq_num != 0:
                data_seq_list = np.array(data_seq_list,dtype=np.float64)
                zero_padding = np.zeros((data_seq_length - seq_num, 9),dtype=np.float64)
                label_padding = np.zeros((1, 9)) + int(last_label)
                data_seq_list = np.concatenate((data_seq_list, zero_padding,label_padding), axis=0)
                data_track_list.append(data_seq_list)
                data_seq_list = list()
            seq_num = 0
        line[-1] = last_label
        alpha,r,height,v,time,rcs,label = line
        fuyangjiao = np.degrees(np.arcsin(height/r))
        data_seq_list.append([time,r,alpha,fuyangjiao,v,seq_num,height,rcs,label])
        seq_num += 1
    data_seq_list = np.array(data_seq_list,dtype=np.float64)
    zero_padding = np.zeros((data_seq_length - seq_num, 9),dtype=np.float64)
    label_padding = np.zeros((1, 9)) + int(last_label)
    data_seq_list = np.concatenate((data_seq_list, zero_padding,label_padding), axis=0)
    data_track_list.append(data_seq_list)
    data_track_list = np.array(data_track_list,dtype=np.float64)
    print(f"processed data shape: {data_track_list.shape}")
    np.save(processed_data_path2, data_track_list)

def dataset_split(dataset_root : str, data_path : str, test_ratio : float = 0.2, val_ratio : float = 0.2)-> None:
    """
    split dataset into train, test and val set

    Input:
    :param dataset_root: dataset root path
    :param data_path: processed data path
    :param test_ratio: test set ratio
    :param val_ratio: val set ratio
    :return: None

    """
    total_data = np.load(data_path)
    cls0 = total_data[total_data[:,-1,0]==0]
    cls1 = total_data[total_data[:,-1,0]==1]
    cls0_train = cls0[:int((1-test_ratio-val_ratio)*cls0.shape[0])]
    cls0_test = cls0[int((1-test_ratio-val_ratio)*cls0.shape[0]):int((1-val_ratio)*cls0.shape[0])]
    cls0_val = cls0[int((1-val_ratio)*cls0.shape[0]):]
    cls1_train = cls1[:int((1-test_ratio-val_ratio)*cls1.shape[0])]
    cls1_test = cls1[int((1-test_ratio-val_ratio)*cls1.shape[0]):int((1-val_ratio)*cls1.shape[0])]
    cls1_val = cls1[int((1-val_ratio)*cls1.shape[0]):]
    train_data = np.concatenate((cls0_train,cls1_train),axis=0)
    test_data = np.concatenate((cls0_test,cls1_test),axis=0)
    val_data = np.concatenate((cls0_val,cls1_val),axis=0)
    print(f"train data shape: {train_data.shape}")
    print(f"test data shape: {test_data.shape}")
    print(f"val data shape: {val_data.shape}")
    np.save(dataset_root + "/train/train.npy", train_data)
    np.save(dataset_root + "/test/test.npy", test_data)
    np.save(dataset_root + "/eval/val.npy", val_data)

if __name__ == '__main__':

    # raw_data_path = "raw_data/raw_data/track2_droneRegconition.xlsx"
    # xlsx2csv(raw_data_path)
    # csv_data_path = "raw_data/track2_droneRegconition.csv"
    # csv2npy(csv_data_path)
    # file = np.load("raw_data/track2_droneRegconition.npy")
    # print(file[:1])
    dataset_split("pre_expriment","raw_data/track2_droneRegconition.npy",0.2,0.2)