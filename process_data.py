import math
import random
import process_logs


def get_positive_tap_samples():
    """
    Returns lists of sensor values corresponding to a 300ms window containing
    the largest absolute acceleration value from each tap.
    :return:
    """
    pass


def get_negative_tap_samples():
    pass


def get_highest_lines(data_list):
    """
    Returns the log lines where the sensor values register the highest x-y
    magnitude.
    E.g. if there are 15 data windows in the data list, then this function
    returns the 15 lines containing sensor values with the highest magnitudes.
    :param data_list: A list of data windows generated from the process_...
    functions.
    :return: The log line containing sensor values of highest magnitude.
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


def make_hand_data_unscaled(file_name):
    """
    Creates the left/right hand training/testing file in the format:
    <hand[-1 for left, +1 for right]> <index1>:<angle at max magnitude> ...
    :param file_name: Name of training file to be written
    :return:
    """
    # Get lists of data windows for left/right hand taps
    [lhand_left_taps, lhand_right_taps] = \
        process_logs.process_2p_left_hand_lin_acc()
    [rhand_left_taps, rhand_right_taps] = \
        process_logs.process_2p_right_hand_lin_acc()
    # Get log lines with highest x/y acceleration for left/right hand
    lhand_highest_log_lines = [get_highest_lines(lhand_left_taps),
                               get_highest_lines(lhand_right_taps)]
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

    # Shuffling?
    random.shuffle(left_hand_angles)
    random.shuffle(right_hand_angles)

    # Write to training file
    file = open("training/" + file_name + ".train", 'w', encoding='utf-8')
    for angle_sample in left_hand_angles:
        file.write("-1 1:" + str(angle_sample) + '\n')
    for angle_sample in right_hand_angles:
        file.write("+1 1:" + str(angle_sample) + '\n')
    file.close()


# make_hand_data_unscaled("hand_2p_unscaled")
