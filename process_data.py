import json
import os
import math
import random


def process_2p_left_hand_lin_acc():
    """
    Sorts data logged for left hand, linear accelerometer, for 2 tap points
    :return: 2 lists containing data for left/right taps
    """
    log_dir = os.path.join('logs', '2_points', 'left_hand', 'lin_acc')
    file_names = os.listdir(log_dir)  # list of strings, '1', '2', etc

    left_taps = []
    right_taps = []

    for file_name in file_names:
        file = open(os.path.join(log_dir, file_name), 'r+', encoding='utf-8')
        content = file.read()
        file.close()

        split_lines = content.splitlines()
        data_window = []

        for i in range(len(split_lines)):
            if i > 0:  # Start at the second line
                split_line = split_lines[i].split(",")
                split_line_prev = split_lines[i - 1].split(",")
                if int(split_line[1]) == int(split_line_prev[1]):
                    data_window.append(split_lines[i])
                elif int(split_line[1]) == 2 and int(split_line_prev[1]) == 1:
                    left_taps.append(data_window.copy())
                    data_window.clear()
                    data_window.append(split_lines[i])
                elif int(split_line[1]) == 1 and int(split_line_prev[1]) == 2:
                    right_taps.append(data_window.copy())
                    data_window.clear()
                    data_window.append(split_lines[i])
                if i == (len(split_lines) - 1):  # Last line
                    right_taps.append(data_window.copy())

    return [left_taps, right_taps]


def process_2p_right_hand_lin_acc():
    """
    Sorts data logged for right hand, linear accelerometer, for 2 tap points
    :return: 2 lists containing data for left/right taps
    """
    log_dir = os.path.join('logs', '2_points', 'right_hand', 'lin_acc')
    file_names = os.listdir(log_dir)  # list of strings, '1', '2', etc

    left_taps = []
    right_taps = []

    for file_name in file_names:
        file = open(os.path.join(log_dir, file_name), 'r+', encoding='utf-8')
        content = file.read()
        file.close()

        split_lines = content.splitlines()
        data_window = []

        for i in range(len(split_lines)):
            if i > 0:  # Start at the second line
                split_line = split_lines[i].split(",")
                split_line_prev = split_lines[i - 1].split(",")
                if int(split_line[1]) == int(split_line_prev[1]):
                    data_window.append(split_lines[i])
                elif int(split_line[1]) == 2 and int(split_line_prev[1]) == 1:
                    left_taps.append(data_window.copy())
                    data_window.clear()
                    data_window.append(split_lines[i])
                elif int(split_line[1]) == 1 and int(split_line_prev[1]) == 2:
                    right_taps.append(data_window.copy())
                    data_window.clear()
                    data_window.append(split_lines[i])
                if i == (len(split_lines) - 1):  # Last line
                    right_taps.append(data_window.copy())

    return [left_taps, right_taps]


def get_highest_lines(data_list):
    """
    Returns the log lines where the sensor values register the highest x-y
    magnitude.
    E.g. if there are 15 data windows in the data list, then this function
    returns the 15 lines containing sensor values with the highest magnitudes.
    :param data_list: A list of data windows
    :return: The log line containing sensor values of highest magnitude
    """
    max_magnitude_lines = []
    for data_window in data_list:
        max_magnitude_line = ""
        max_magnitude = 0
        for line in data_window:
            split_line = line.split(",")
            x_value = float(split_line[2])
            y_value = float(split_line[3])
            magnitude = math.sqrt(x_value**2 + y_value**2)
            if max_magnitude < magnitude:
                max_magnitude = magnitude
                max_magnitude_line = line
        max_magnitude_lines.append(max_magnitude_line)
    return max_magnitude_lines


def get_angle(log_line):
    """
    Calculates angle in the x-y plane for the entry with the highest magnitude
    of sensor values
    :param log_line: String containing a line of sensor log data
    :return: Angle made by highest magnitude impact, in radians. Ranges from
    -pi to pi
    """
    split_log_line = log_line.split(",")
    x_value = float(split_log_line[2])
    y_value = float(split_log_line[3])
    return math.atan2(y_value, x_value)


def make_hand_data_file(file_name):
    """
    Creates the left/right hand training/testing file in the format:
    <hand[-1 for left, +1 for right]> <index1>:<angle at max magnitude> ...
    :param file_name: Name of training file to be written
    :return:
    """
    # Get log data containing highest sensor magnitudes
    [lhand_left_taps, lhand_right_taps] = process_2p_left_hand_lin_acc()
    lhand_highest_log_lines = [get_highest_lines(lhand_left_taps),
                               get_highest_lines(lhand_right_taps)]
    [rhand_left_taps, rhand_right_taps] = process_2p_right_hand_lin_acc()
    rhand_highest_log_lines = [get_highest_lines(rhand_left_taps),
                               get_highest_lines(rhand_right_taps)]
    # Get angles
    left_hand_angles = []
    right_hand_angles = []
    for tap_position in lhand_highest_log_lines:
        for log_line in tap_position:
            left_hand_angles.append(get_angle(log_line))
    for tap_position in rhand_highest_log_lines:
        for log_line in tap_position:
            right_hand_angles.append(get_angle(log_line))
    # Scale to [-1,1]
    max_angle = max(left_hand_angles + right_hand_angles)
    min_angle = min(left_hand_angles + right_hand_angles)
    left_hand_angles_scaled = [2 * (angle - min_angle) /
                               (max_angle - min_angle) -
                               1 for angle in left_hand_angles]
    right_hand_angles_scaled = [2 * (angle - min_angle) /
                                (max_angle - min_angle) -
                                1 for angle in right_hand_angles]
    # Shuffling done in place
    random.shuffle(left_hand_angles_scaled)
    random.shuffle(right_hand_angles_scaled)
    # Get sample sizes for left and right hand testing and training data
    left_hand_train_sample_size = int(len(left_hand_angles_scaled) * 0.9)
    right_hand_train_sample_size = int(len(right_hand_angles_scaled) * 0.9)
    # Write to training file
    file = open("training/" + file_name + ".train", 'w', encoding='utf-8')
    for i in range(left_hand_train_sample_size):
        file.write("-1 1:" + str(left_hand_angles_scaled[i]) + '\n')
    for i in range(right_hand_train_sample_size):
        file.write("+1 1:" + str(right_hand_angles_scaled[i]) + '\n')
    file.close()
    # Write to testing file
    file = open("training/" + file_name + ".test", 'w', encoding='utf-8')
    for i in range(left_hand_train_sample_size, len(left_hand_angles_scaled)):
        file.write("-1 1:" + str(left_hand_angles_scaled[i]) + '\n')
    for i in range(right_hand_train_sample_size, len(right_hand_angles_scaled)):
        file.write("+1 1:" + str(right_hand_angles_scaled[i]) + '\n')
    file.close()


make_hand_data_file("hand_2p")
