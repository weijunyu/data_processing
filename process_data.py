import math
import random
import process_logs
import statistics
import pprint


def get_left_hand_positive_tap_samples():
    """
    Returns lists of sensor values corresponding to a 300ms window containing
    the largest absolute acceleration value from each tap.
    :return: A list of positive samples for each tap point. Each positive sample
    contains 30 data points starting from the moment the highest x-y L2-norm
    values are captured.
    """
    # Get sensor data for all tap locations
    raw_logs = process_logs.process_5p_left_hand_lin_acc()
    positive_samples = []
    # Get indices of all the highest sensor magnitude log entries.
    for tap_number in range(len(raw_logs)):  # Tap locations 0 - 4
        current_location_log = raw_logs[tap_number]
        if tap_number == len(raw_logs) - 1:  # Next point is 0 if current is 4
            next_location_log = raw_logs[0]
        else:
            next_location_log = raw_logs[tap_number + 1]

        cur_location_samples = []  # Holds all samples for current tap location
        highest_log_indices = []
        max_magnitude_lines = get_highest_lines(current_location_log)

        for line in max_magnitude_lines:
            # for data_window in current_location_log:
            #     if line in data_window:
            #         highest_log_indices.append(data_window.index(line))
            [highest_log_indices.append(data_window.index(line)) for
             data_window in current_location_log if line in data_window]

        # Get positive samples as highest value log + the next 29 log entries
        last_entry_indices = [index + 30 for index in highest_log_indices]
        # For each data window for current tap location
        for i in range(len(current_location_log)):
            # If the positive sample requirement exceeds the data window
            if last_entry_indices[i] > len(current_location_log[i]):
                first_half_sample = current_location_log[i][highest_log_indices[i]:last_entry_indices[i]]
                # If point 5, the second half of sample comes from point 1 in the next log
                # TODO: Potential bug where the last entry for point 5 has no next log at location 1 (Very unlikely)
                if tap_number == len(raw_logs) - 1:
                    second_half_sample = next_location_log[i + 1][:last_entry_indices[i] - len(current_location_log[i])]
                else:
                    second_half_sample = next_location_log[i][:(last_entry_indices[i] - len(current_location_log[i]))]
                full_sample = first_half_sample + second_half_sample
                cur_location_samples.append(full_sample)
            else:
                cur_location_samples.append(current_location_log[i][highest_log_indices[i]:last_entry_indices[i]])
        positive_samples.append(cur_location_samples)
    # Simple check to ensure each positive sample contains 30 data points.
    # print(len(positive_samples))
    # for tap_location in positive_samples:
    #     print(len(tap_location))
    #     print(statistics.mean([len(sample) for sample in tap_location]))
    return positive_samples


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
get_positive_tap_samples()
