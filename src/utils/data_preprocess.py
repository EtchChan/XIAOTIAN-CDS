"""
Brief: raw data preprocessing functions set
Author: alfeak, CHEN Yi-xuan
updateDate: 2024-10-13
"""

import pandas as pd
import numpy as np
import os
import csv
import matplotlib.pyplot as plt
import glob

# from src.ExtendedExperiment.tvm_infer import output

"""
/brief: padding the data to a fixed length.

/author: CHEN Yi-xuan

/date: 2024-09-24

/param: data: the data to be padded
        fixed_length: the length to be padded to, should be greater than the length of the data
"""
def padding_data_to_fixed_length(data, fixed_length: int):
    # check if the fixed length is greater than the length of the data
    if fixed_length < len(data):
        raise ValueError("The fixed length should be greater than the length of the data.")
    # get the dimension of each data point
    dim = len(data[0])
    # padding the data to the fixed length
    padding = [[0.0 for _ in range(dim)] for _ in range(fixed_length - len(data))]
    data.extend(padding)

    return data


"""
/brief: convert polar coordinate to Cartesian coordinate.

/author: CHEN Yi-xuan

/date: 2024-10-13
"""
def spherical_to_cartesian(r, azimuth, elevation):
    """
    convert polar coordinate to Cartesian coordinate

    Input:
    :param r: radial component(slant range or radial velocity)
    :param azimuth: azimuth angle
    :param elevation: elevation angle
    :return: x, y, z
    """
    azimuth_rad = np.radians(azimuth)
    elevation_rad = np.radians(elevation)
    x = r * np.cos(elevation_rad) * np.cos(azimuth_rad)
    y = r * np.cos(elevation_rad) * np.sin(azimuth_rad)
    z = r * np.sin(elevation_rad)
    return x, y, z


def extract_event_1_data_from_csv(raw_data_folder: str, fixed_length: int = -1):
    """
    extract event_1 data from a CSV file and process it into a numpy array.

    :param raw_data_folder: str, the folder contains the UTF-8 CSV track files
    :param fixed_length: int, the length to be padded to, should be greater than the length of the data
    :return: np.ndarray, the processed data
    """
    # Check if the raw data folder exists
    if not os.path.exists(raw_data_folder):
        raise FileNotFoundError(f"Raw data folder {raw_data_folder} not found")
    # Get the list of CSV files in the raw data folder
    raw_data_files = glob.glob(os.path.join(raw_data_folder, 'raw_tracks_*.csv'))

    database_cnt = 1
    for raw_data_file in raw_data_files:
        # Read CSV file, skipping the header
        df = pd.read_csv(raw_data_file, skiprows=1,
                         names=['Time', 'Slant_range', 'Azimuth_angle', 'Elevation_angle', 'Radial_velocity',
                                'Num_orbits'])

        # Convert spherical coordinates to Cartesian
        x, y, z = spherical_to_cartesian(df['Slant_range'], df['Azimuth_angle'], df['Elevation_angle'])

        # Calculate radial velocity components
        v_x, v_y, v_z = spherical_to_cartesian(df['Radial_velocity'], df['Azimuth_angle'], df['Elevation_angle'])

        # Create new DataFrame with converted data
        new_df = pd.DataFrame({
            'Time': df['Time'],
            'x': x,
            'y': y,
            'z': z,
            'v_radial_x': v_x,
            'v_radial_y': v_y,
            'v_radial_z': v_z,
            'Slant_range': df['Slant_range'],
            'Azimuth_angle': df['Azimuth_angle'],
            'Elevation_angle': df['Elevation_angle'],
            'Radial_velocity': df['Radial_velocity']
        })

        # save the processed data to a npy file under the same directory as the raw csv data
        output_file = os.path.splitext(raw_data_file)[0] + '.npy'
        print(f"Processed {raw_data_file} and saved as {output_file}; total radar points: {len(new_df)}")
        np.save(output_file, new_df)


        # # Group data by revolutions
        # grouped = new_df.groupby(df['Num_orbits'])
        #
        # # Create list to store data grouped by revolutions
        # radar_image_data = []  # radar data points in one orbit/revolution
        #                   # shape: [L, 11], contains 11 features after conversion
        #
        # # Process each revolution
        # for _, group in grouped:
        #     radar_image_data.append(group.values)
        #
        # # print the statistics of the number of radar points in each revolution if it is not padded
        # if fixed_length < 0:
        #     print(f"min points nums in a radar image: {min([len(x) for x in radar_image_data])}")
        #     print(f"max points nums in a radar image: {max([len(x) for x in radar_image_data])}")
        #     print(f"mean points nums in a radar image: {np.mean([len(x) for x in radar_image_data])}")
        #     # plot a histogram of the number of radar points in each revolution and mark the mean value on it
        #     plt.hist([len(x) for x in radar_image_data], bins=60)
        #     plt.axvline(x=np.mean([len(x) for x in radar_image_data]), color='r', linestyle='--')
        #     plt.text(plt.xlim()[1], plt.ylim()[1] - 1, f'Mean: {np.mean([len(x) for x in radar_image_data]):.2f}',
        #              horizontalalignment='right', verticalalignment='top')
        #     # mark the total number of revolutions on the plot
        #     plt.text(plt.xlim()[1], plt.ylim()[1] - 5, f"Total Revolutions: {len(radar_image_data)}",
        #              horizontalalignment='right', verticalalignment='top')
        #     plt.title(f"Event 1 database {database_cnt}: Distribution of radar-point-nums in each revolution")
        #     fig_path = os.path.splitext(raw_data_file)[0] + '_radar_points_num_distribution_event_1.png'
        #     plt.tight_layout()
        #     plt.savefig(fig_path)
        #     plt.show()
        #
        #
        # # Save data as NumPy file
        # output_file = os.path.splitext(raw_data_file)[0] + '.npy'
        # np.save(output_file, np.array(radar_image_data, dtype=object))
        #
        # print(f"Processed {raw_data_file} and saved as {output_file}, total revolutions: {len(radar_image_data)}")
        # database_cnt += 1


"""
/brief: extract event_2 data from a CSV file and process it into a numpy array.

/author: CHEN Yi-xuan

/date: 2024-09-24

/param: file_path: str, the path of the UTF-8 CSV file
        fixed_length: int, the length to be padded to, should be greater than the length of the data

/input: the raw format is as follows:
    azimuth angle, slant range, relative height, radial velocity, record time, RCS, label  # start of track k
     azimuth angle, slant range, relative height, radial velocity, record time, RCS,
        ...
     azimuth angle, slant range, relative height, radial velocity, record time, RCS,       # end of track k
    azimuth angle, slant range, relative height, radial velocity, record time, RCS, label  # start of track k+1
        ...
    only the first row has the label, the rest of the rows have an empty label.
    Because radar points in a track are continuous and have the same label.
    
/output: the processed data is as follows:
    [N, [[L, 6], 1, 1]] i.e. [N, [track, label, L]]
    N is the number of tracks;
    L is the length of a track, i.e. number of radar points in a track;
    (optional: padding L to a fixed length, e.g. 15; and remove the length of the track;)
    6 is the number of features of a radar point;
    first 1 is the label of the track, it is either 0 or 1, 0 for non-drone, 1 for drone.
    second 1 is the length of the track, it is the same as L.
"""
def extract_event_2_data_from_csv(raw_data_path: str, fixed_length: int = -1):
    if -1 < fixed_length < 11:
        raise ValueError("The fixed length should be negative(no padding) or no less than 11(max_len_of_track).")
    data_track_list = [] # N * ([L, 6] + [1] + [1]), record the data of N tracks
    track = [] # [N, 6], record the data of a track
    last_label = -1 # keep the label of the last track
    cnt = 0 # count the number of radar points in last track
    num_of_tracks = 0 # count the number of tracks

    with open(raw_data_path, 'r', encoding="utf-8") as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header

        for line in reader:
            if line[6].strip():  # Check if the 7th column (label) is not empty
                if last_label != -1:
                    # Add previous track to the data_track_list
                    if fixed_length < 0:
                        data_track_list.append([track, last_label, cnt])
                    else:
                        # Padding the track to fixed length
                        track = padding_data_to_fixed_length(track, fixed_length)
                        data_track_list.append([track, last_label])
                    track = []
                cnt = 1
                last_label = int(line[6])
                track.append([float(x) for x in line[:6]]) # add the first radar point to the track
                num_of_tracks += 1
            else:
                track.append([float(x) for x in line[:6]]) # Take only the first 6 columns as radar point of the track
                cnt += 1 # count the number of radar points in a track

        # Add the last track to the data_track_list
        if fixed_length < 0:
            data_track_list.append([track, last_label, cnt])
        else:
            if cnt < fixed_length:
                track = padding_data_to_fixed_length(track, fixed_length)
            data_track_list.append([track, last_label])

    # calculate the statistics of the length of tracks
    print(f"track total nums: {num_of_tracks}")
    if fixed_length < 0:
        track_length = [x[2] for x in data_track_list]
        mean_track_length = np.mean(track_length)
        print(f"track min points nums: {min(track_length)}")
        print(f"track max points nums: {max(track_length)}")
        print(f"track mean points nums: {mean_track_length}")
        # plot a histogram of the length of tracks and mark the mean value on it
        plt.hist(track_length, bins=20)
        plt.axvline(x=mean_track_length, color='r', linestyle='--')
        plt.text(plt.xlim()[1], plt.ylim()[1], f'Mean: {mean_track_length:.2f}',
                 horizontalalignment='right', verticalalignment='top')
        plt.title("Event 2: Distribution of the length of tracks")
        plt.tight_layout()
        plt.savefig("./event_2_track_length_distribution.png")
        plt.show()

    # show the statistics of the label
    label_list = [x[1] for x in data_track_list]
    counts = [label_list.count(0), label_list.count(1)]
    print(f"not-drone-instance nums: {counts[0]}")
    print(f"drone-instance nums: {counts[1]}")
    # plot a bar chart of the distribution of type of instances
    labels = ['not-drone', 'drone']
    fig, ax = plt.subplots()
    bars = ax.bar([0, 1], counts, tick_label=labels)

    # Add labels on top of each bar
    for i, v in enumerate(counts):
        ax.text(i, v, str(v), ha='center', va='bottom')

    plt.title("Event 2: Distribution of the type of instances")
    plt.tight_layout()
    plt.savefig("./event_2_data_distribution.png")
    plt.show()

    # process data to npy format and print its shape info
    data_track_list = np.array(data_track_list, dtype=object)
    print(f"Processed data shape: {len(data_track_list)}")

    # save the processed data to a npy file under the same directory as the raw csv data
    if fixed_length < 0: # set the name as raw_tracks_graph.npy under the same directory
        output_path = raw_data_path.replace('.csv', '_tracks_graph.npy')
        np.save(output_path, data_track_list)
    else: # set the name as raw_data_padded.npy under the same directory
        output_path = raw_data_path.replace('.csv', '_padded.npy')
        np.save(output_path, data_track_list)

    return output_path


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
    # dataset_split("pre_expriment","raw_data/track2_droneRegconition.npy",0.2,0.2)

    # usage of extract_event_2_data_from_csv
    # extract_event_2_data_from_csv("../../data/event_2/raw_data.csv")

    # usage of extract_event_1_data_from_csv
    extract_event_1_data_from_csv("../../data/event_1/")