import math
import random
import process_logs
import statistics
import pprint
import numpy

from scipy import stats
from numpy import matrix
from numpy import linalg

LEFT_HAND = "left hand"
RIGHT_HAND = "right hand"
LINEAR_ACCELEROMETER = "linear accelerometer"
GYROSCOPE = "gyroscope"


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


def get_sample_p2p(tap_location_samples):
    """
    :param tap_location_samples: A list of positive or negative samples for a
    particular tap location.
    :return: A list of x, y, and z axis peak to peak values for each sample,
    corresponding to the given tap location.
    """
    p2p_values = []
    for sample in tap_location_samples:
        x_max = max([float(log_line.split(",")[2]) for log_line in sample])
        y_max = max([float(log_line.split(",")[3]) for log_line in sample])
        z_max = max([float(log_line.split(",")[4]) for log_line in sample])
        x_min = min([float(log_line.split(",")[2]) for log_line in sample])
        y_min = min([float(log_line.split(",")[3]) for log_line in sample])
        z_min = min([float(log_line.split(",")[4]) for log_line in sample])

        x_p2p = abs(x_max - x_min)
        y_p2p = abs(y_max - y_min)
        z_p2p = abs(z_max - z_min)

        p2p_values.append([x_p2p, y_p2p, z_p2p])
    return p2p_values


def get_peak_value_sign(tap_location_samples):
    """
    Gets the sign of the peak sensor values on the x, y, and z axes.
    :param tap_location_samples:
    :return:
    """
    peak_value_signs = []
    for sample in tap_location_samples:
        x_values = [float(log_line.split(",")[2]) for log_line in sample]
        y_values = [float(log_line.split(",")[3]) for log_line in sample]
        z_values = [float(log_line.split(",")[4]) for log_line in sample]
        x_values_abs = [abs(value) for value in x_values]
        y_values_abs = [abs(value) for value in y_values]
        z_values_abs = [abs(value) for value in z_values]
        x_max = max(x_values_abs)
        y_max = max(y_values_abs)
        z_max = max(z_values_abs)
        x_max_index = x_values_abs.index(x_max)
        y_max_index = y_values_abs.index(y_max)
        z_max_index = z_values_abs.index(z_max)
        x_max_sign = numpy.sign(x_values[x_max_index])
        y_max_sign = numpy.sign(y_values[y_max_index])
        z_max_sign = numpy.sign(z_values[z_max_index])
        peak_value_signs.append([x_max_sign, y_max_sign, z_max_sign])
    return peak_value_signs


def get_sample_mean(tap_location_samples):
    """
    :param tap_location_samples: A list of positive or negative samples for a
    particular tap location.
    :return: A list of x, y, and z axis mean values for each sample,
    corresponding to the given tap location.
    """
    tap_location_means = []
    for sample in tap_location_samples:
        tap_location_means.append(
                [
                    statistics.mean(
                        [abs(float(log_line.split(",")[2]) for log_line in sample)]
                    ),
                    statistics.mean(
                        [abs(float(log_line.split(",")[3]) for log_line in sample)]
                    ),
                    statistics.mean(
                        [abs(float(log_line.split(",")[4]) for log_line in sample)]
                    ),
                ]
        )
    return tap_location_means


def get_sample_median(tap_location_samples):
    """
    :param tap_location_samples: A list of positive or negative samples for a
    particular tap location.
    :return: A list of x, y, and z axis median values for each sample,
    corresponding to the given tap location.
    """
    tap_location_median = []
    for sample in tap_location_samples:
        tap_location_median.append(
            [
                statistics.median(
                        [float(log_line.split(",")[2]) for log_line in sample]
                ),
                statistics.median(
                        [float(log_line.split(",")[3]) for log_line in sample]
                ),
                statistics.median(
                        [float(log_line.split(",")[4]) for log_line in sample]
                ),
            ]
        )
    return tap_location_median


def get_sample_std_dev(tap_location_samples):
    """
    :param tap_location_samples: A list of positive or negative samples for a
    particular tap location.
    :return: A list of x, y, and z axis standard deviation values for each
    sample, corresponding to the given tap location.
    """
    tap_location_std_dev = []
    for sample in tap_location_samples:
        tap_location_std_dev.append(
            [
                statistics.stdev(
                        [float(log_line.split(",")[2]) for log_line in sample]
                ),
                statistics.stdev(
                        [float(log_line.split(",")[3]) for log_line in sample]
                ),
                statistics.stdev(
                        [float(log_line.split(",")[4]) for log_line in sample]
                ),
            ]
        )
    return tap_location_std_dev


def get_sample_skew(tap_location_samples):
    """
    :param tap_location_samples: A list of positive or negative samples for a
    particular tap location.
    :return: A list of x, y, and z axis skewness values for each
    sample, corresponding to the given tap location.
    """
    tap_location_skew = []
    for sample in tap_location_samples:
        tap_location_skew.append(
            [
                stats.skew(
                    [float(log_line.split(",")[2]) for log_line in sample]
                ),
                stats.skew(
                    [float(log_line.split(",")[3]) for log_line in sample]
                ),
                stats.skew(
                    [float(log_line.split(",")[4]) for log_line in sample]
                ),
            ]
        )
    return tap_location_skew


def get_sample_kurtosis(tap_location_samples):
    """
    :param tap_location_samples: A list of positive or negative samples for a
    particular tap location.
    :return: A list of x, y, and z axis kurtosis values for each
    sample, corresponding to the given tap location.
    """
    tap_location_kurtosis = []
    for sample in tap_location_samples:
        tap_location_kurtosis.append(
            [
                stats.kurtosis(
                        [float(log_line.split(",")[2]) for log_line in sample]
                ),
                stats.kurtosis(
                        [float(log_line.split(",")[3]) for log_line in sample]
                ),
                stats.kurtosis(
                        [float(log_line.split(",")[4]) for log_line in sample]
                ),
            ]
        )
    return tap_location_kurtosis


def get_l1_norm(tap_location_samples):
    l1_norms = []
    for sample in tap_location_samples:
        corr_matrix = matrix(
            [
                [float(log_line.split(",")[2]),
                 float(log_line.split(",")[3]),
                 float(log_line.split(",")[4])]
                for log_line in sample
            ]
        )
        l1_norms.append(linalg.norm(corr_matrix, ord=1))
    return l1_norms


def get_inf_norm(tap_location_samples):
    inf_norms = []
    for sample in tap_location_samples:
        corr_matrix = matrix(
            [
                [float(log_line.split(",")[2]),
                 float(log_line.split(",")[3]),
                 float(log_line.split(",")[4])]
                for log_line in sample
            ]
        )
        inf_norms.append(linalg.norm(corr_matrix, ord=numpy.inf))
    return inf_norms


def get_fro_norm(tap_location_samples):
    """
    :param tap_location_samples: A list of positive or negative samples for a
    particular tap location.
    :return: A list of Frobenius norms (corresponding to each sample) applied to
    a matrix containing x, y, and z axis sensor values in each column.
    """
    fro_norms = []
    for sample in tap_location_samples:
        corr_matrix = matrix(
            [
                [float(log_line.split(",")[2]),
                 float(log_line.split(",")[3]),
                 float(log_line.split(",")[4])]
                for log_line in sample
            ]
        )
        fro_norms.append(linalg.norm(corr_matrix, ord='fro'))
    return fro_norms


def get_pearson_coeff(lin_acc_samples, gyro_samples):
    """
    :param lin_acc_samples:
    :param gyro_samples:
    :return: A list of x, y, and z axis pearson values, for each sample pair.
    """
    # Get array of x, y, and z linear acceleration values
    lin_acc_container = []
    for sample in lin_acc_samples:
        lin_acc_x = [float(log_line.split(",")[2]) for log_line in sample]
        lin_acc_y = [float(log_line.split(",")[3]) for log_line in sample]
        lin_acc_z = [float(log_line.split(",")[4]) for log_line in sample]
        lin_acc_container.append([lin_acc_x, lin_acc_y, lin_acc_z])
    gyro_container = []
    for sample in gyro_samples:
        gyro_x = [float(log_line.split(",")[2]) for log_line in sample]
        gyro_y = [float(log_line.split(",")[3]) for log_line in sample]
        gyro_z = [float(log_line.split(",")[4]) for log_line in sample]
        gyro_container.append([gyro_x, gyro_y, gyro_z])
    pearson_coeff = []
    for i in range(len(lin_acc_container)):  # The two containers have same len
        pearson_axes = []
        for j in range(len(lin_acc_container[i])):
            pearson_axes.append(
                stats.pearsonr(lin_acc_container[i][j], gyro_container[i][j])[0]
            )
        pearson_coeff.append(pearson_axes)
    return pearson_coeff


def get_positive_tap_samples(sensor, hand):
    """
    Returns lists of sensor values corresponding to a 300ms window containing
    the largest absolute acceleration value from each tap.
    :param sensor: Which sensor the data corresponds to.
    :param hand: Which holding hand the data corresponds to.
    :return: A list of positive samples for each tap point. Each positive sample
    contains 15 data points (15 * 20ms = 300ms) starting from the moment the
    highest x-y L2-norm values are captured.
    """
    # Get sensor data for all tap locations
    if sensor == LINEAR_ACCELEROMETER:
        if hand == LEFT_HAND:
            raw_logs = process_logs.process_5p_left_hand_lin_acc()
        elif hand == RIGHT_HAND:
            raw_logs = process_logs.process_5p_right_hand_lin_acc()
        else:
            print("Please use a valid hand position")
            return "Invalid hand position"
    elif sensor == GYROSCOPE:
        if hand == LEFT_HAND:
            raw_logs = process_logs.process_5p_left_hand_gyroscope()
        elif hand == RIGHT_HAND:
            raw_logs = process_logs.process_5p_right_hand_gyroscope()
        else:
            print("Please use a valid hand position")
            return "Invalid hand position"
    else:
        print("Sensor type invalid.")
        return "Invalid sensor type."

    positive_samples = []
    for tap_number in range(len(raw_logs)):  # Tap locations 0 - 4
        current_location_log = raw_logs[tap_number]
        if tap_number == len(raw_logs) - 1:  # Next point is 0 if current is 4
            next_location_log = raw_logs[0]
        else:
            next_location_log = raw_logs[tap_number + 1]

        cur_location_samples = []  # Holds all samples for current tap location
        highest_log_indices = []
        max_magnitude_lines = get_highest_lines(current_location_log)

        # Get indices of all the highest sensor magnitude log entries.
        for line in max_magnitude_lines:
            [highest_log_indices.append(data_window.index(line)) for
             data_window in current_location_log if line in data_window]

        # Get positive samples as highest value log + the next 29 log entries
        last_entry_indices = [index + 15 for index in highest_log_indices]
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

    # Simple check to ensure each positive sample contains 15 data points.
    # print(len(positive_samples))
    # for tap_location in positive_samples:
    #     print(len(tap_location))
    #     print(statistics.mean([len(sample) for sample in tap_location]))

    return positive_samples


def get_negative_tap_samples(sensor, hand):
    """
    Returns lists of sensor values corresponding to a 300ms window that is
    a distance away from a positive sample.
    :param sensor: Which sensor the data corresponds to.
    :param hand: Which holding hand the data corresponds to.
    :return: A list of positive samples for each tap point. Each positive sample
    contains 15 data points (15 * 20ms = 300ms) starting from the moment the
    highest x-y L2-norm values are captured.
    """
    # Get sensor data for all tap locations
    if sensor == LINEAR_ACCELEROMETER:
        if hand == LEFT_HAND:
            raw_logs = process_logs.process_5p_left_hand_lin_acc()
        elif hand == RIGHT_HAND:
            raw_logs = process_logs.process_5p_right_hand_lin_acc()
        else:
            print("Please use a valid hand position")
            return "Invalid hand position"
    elif sensor == GYROSCOPE:
        if hand == LEFT_HAND:
            raw_logs = process_logs.process_5p_left_hand_gyroscope()
        elif hand == RIGHT_HAND:
            raw_logs = process_logs.process_5p_right_hand_gyroscope()
        else:
            print("Please use a valid hand position")
            return "Invalid hand position"
    else:
        print("Sensor type invalid.")
        return "Invalid sensor type."

    negative_samples = []
    for tap_number in range(len(raw_logs)):  # Tap locations 0 - 4
        current_location_log = raw_logs[tap_number]

        # Get indices of all the highest sensor magnitude log entries.
        highest_log_indices = []
        max_magnitude_lines = get_highest_lines(current_location_log)
        for line in max_magnitude_lines:
            [highest_log_indices.append(data_window.index(line)) for
             data_window in current_location_log if line in data_window]

        cur_location_samples = []  # Holds all samples for current tap location
        for i in range(len(current_location_log)):
            # For each data window
            if highest_log_indices[i] < len(current_location_log[i]) - \
                    15 - 10 - 15:
                starting_index = highest_log_indices[i] + 15 + 10
                ending_index = starting_index + 15
                sample = \
                    current_location_log[i][starting_index:ending_index]
                cur_location_samples.append(sample)
            else:
                starting_index = highest_log_indices[i] - 10 - 15
                ending_index = starting_index + 15
                sample = \
                    current_location_log[i][starting_index:ending_index]
                cur_location_samples.append(sample)
        negative_samples.append(cur_location_samples)

    # Simple check to ensure each positive sample contains 15 data points.
    # print(len(negative_samples))
    # for tap_location in negative_samples:
    #     print(len(tap_location))
    #     print(statistics.mean([len(sample) for sample in tap_location]))

    return negative_samples


def make_tap_occurrence_data_unscaled(file_name):
    """
    Creates the data file for tap occurrences in the format:
    <[-1 for no tap or +1 for tap] <index1>:<x axis lin acc mean> ...
    :param file_name: Name of training file to be written
    :return:
    """
    # Get positive and negative samples
    lhand_p_samples_lin_acc = get_positive_tap_samples(LINEAR_ACCELEROMETER,
                                                       LEFT_HAND)
    lhand_p_samples_gyro = get_positive_tap_samples(GYROSCOPE, LEFT_HAND)
    rhand_p_samples_lin_acc = get_positive_tap_samples(LINEAR_ACCELEROMETER,
                                                       LEFT_HAND)
    rhand_p_samples_gyro = get_positive_tap_samples(GYROSCOPE, RIGHT_HAND)

    



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


# lhand_pos_gyro = get_positive_tap_samples(GYROSCOPE, LEFT_HAND)
# lhand_pos_lin_acc = get_positive_tap_samples(LINEAR_ACCELEROMETER, LEFT_HAND)
# pprint.pprint(lhand_pos_samples)
# pprint.pprint(get_sample_kurtosis(lhand_pos_samples[0]))
# pprint.pprint(get_inf_norm(lhand_pos_samples[0]))
# pprint.pprint(get_fro_norm(lhand_pos_samples[0]))
# pprint.pprint(get_pearson_coeff(lhand_pos_lin_acc[0], lhand_pos_gyro[0]))
# pprint.pprint(get_sample_p2p(lhand_pos_gyro[0]))
# pprint.pprint(get_peak_value_sign(lhand_pos_gyro[0]))
