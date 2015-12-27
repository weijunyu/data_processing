import json
import os
import math


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


def make_hand_training_file(file_name):
    """
    Creates and writes to the left/right hand training file in the format:
    <hand[0 for left, 1 for right]> <index1>:<angle at max magnitude> ...
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
    # Scale to [-1,1]
    print(json.dumps(lhand_highest_log_lines, indent=2))

    # file = open(file_name, 'w', encoding='utf-8')
    # for tap_position in lhand_highest_log_lines:
    #     for log_line in tap_position:
    #         file.write("0 1:" + str(get_angle(log_line)) + '\n')
    # for tap_position in rhand_highest_log_lines:
    #     for log_line in tap_position:
    #         file.write("1 1:" + str(get_angle(log_line)) + '\n')
    # file.close()


make_hand_training_file("hand_training_2p")

# [lhand_left_taps, lhand_right_taps] = process_2p_left_hand_lin_acc()
# highest_log_lines = get_highest_lines(lhand_left_taps)
# print("highest magnitude for left taps: ")
# print(highest_log_lines)
# highest_log_lines = get_highest_lines(lhand_right_taps)
# print("highest magnitude for right taps: ")
# print(highest_log_lines)
# print("angle of first entry: ")
# print(get_angle(highest_log_lines[0]))


# print(json.dumps(left_taps, indent=2))


# for i in range(len(split_lines)):
#     if i < 10:
#         print(split_lines[i])

# for line in split_lines:
#     print(line)
# split_line = line.split(",")
# print(split_line)


# file = open('data', 'r+', encoding='utf-8')
#
# file.close()
