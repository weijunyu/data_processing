import math
import process_logs
import statistics
import numpy
import os

from scipy import stats

LEFT_HAND = "left hand"
RIGHT_HAND = "right hand"
LINEAR_ACCELEROMETER = "linear accelerometer"
GYROSCOPE = "gyroscope"


def clean_logs():
    # get length of gyro and lin_acc log files, then cut the top x lines in the
    # lin_acc file

    log_dir_gyro = os.path.join('logs', '5_points', 'right_hand', 'gyro')
    log_dir_lin_acc = os.path.join('logs', '5_points', 'right_hand', 'lin_acc')

    file_names = os.listdir(log_dir_gyro)  # list of strings, '1', '2', etc

    for file_name in file_names:
        with open(os.path.join(log_dir_gyro, file_name),
                  encoding='utf-8') as gyro_file:
            with open(os.path.join(log_dir_lin_acc, file_name),
                      encoding='utf-8') as lin_acc_file:
                gyro_content = gyro_file.readlines()
                lin_acc_content = lin_acc_file.readlines()
                gyro_length = len(gyro_content)
                lin_acc_length = len(lin_acc_content)
                diff = lin_acc_length - gyro_length
                for i in range(diff):
                    lin_acc_content.pop(0)
                with open(os.path.join(log_dir_lin_acc, file_name),
                          mode='w', encoding='utf-8') as lin_acc_file_new:
                    for line in lin_acc_content:
                        lin_acc_file_new.write(line)

    log_dir_gyro = os.path.join('logs', '5_points', 'left_hand', 'gyro')
    log_dir_lin_acc = os.path.join('logs', '5_points', 'left_hand', 'lin_acc')

    file_names = os.listdir(log_dir_gyro)  # list of strings, '1', '2', etc

    for file_name in file_names:
        with open(os.path.join(log_dir_gyro, file_name),
                  encoding='utf-8') as gyro_file:
            with open(os.path.join(log_dir_lin_acc, file_name),
                      encoding='utf-8') as lin_acc_file:
                gyro_content = gyro_file.readlines()
                lin_acc_content = lin_acc_file.readlines()
                gyro_length = len(gyro_content)
                lin_acc_length = len(lin_acc_content)
                diff = lin_acc_length - gyro_length
                for i in range(diff):
                    lin_acc_content.pop(0)
                with open(os.path.join(log_dir_lin_acc, file_name),
                          mode='w', encoding='utf-8') as lin_acc_file_new:
                    for line in lin_acc_content:
                        lin_acc_file_new.write(line)


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


def get_angle(tap_location_samples):
    """
    Calculates angle in the x-y plane for the entry with the highest magnitude
    of linear accelerometer sensor values
    :param tap_location_samples: A list of samples for a particular tap
    location.
    :return: Angle made by highest magnitude impact, in radians. Ranges from
    -pi to pi
    """
    return [math.atan2(float(log_line.split(",")[3]),
                       float(log_line.split(",")[2]))
            for log_line in get_highest_lines(tap_location_samples)]
    # for log_line in max_magnitude_list:
    #     [math.atan2(float(log_line.split(",")[3]),
    #                 float(log_line.split(",")[2]))
    #      for log_line in ]
    # split_log_line = log_line.split(",")
    # x_value = float(split_log_line[2])
    # y_value = float(split_log_line[3])
    # return math.atan2(y_value, x_value)


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
    :param tap_location_samples: A list of positive or negative samples for a
    particular tap location.
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


def get_rms(tap_location_samples):
    rms_values = []
    for sample in tap_location_samples:
        squared_x_values = [
            float(log_line.split(",")[2])**2 for log_line in sample
        ]
        squared_y_values = [
            float(log_line.split(",")[3])**2 for log_line in sample
        ]
        squared_z_values = [
            float(log_line.split(",")[4])**2 for log_line in sample
        ]
        mean_squared_x = statistics.mean(squared_x_values)
        mean_squared_y = statistics.mean(squared_y_values)
        mean_squared_z = statistics.mean(squared_z_values)
        rms_x = math.sqrt(mean_squared_x)
        rms_y = math.sqrt(mean_squared_y)
        rms_z = math.sqrt(mean_squared_z)
        rms_values.append([rms_x, rms_y, rms_z])
    return rms_values


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
                        [abs(float(log_line.split(",")[2])) for log_line in sample]
                    ),
                    statistics.mean(
                        [abs(float(log_line.split(",")[3])) for log_line in sample]
                    ),
                    statistics.mean(
                        [abs(float(log_line.split(",")[4])) for log_line in sample]
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
        corr_matrix = numpy.array(
            [
                [float(log_line.split(",")[2]),
                 float(log_line.split(",")[3]),
                 float(log_line.split(",")[4])]
                for log_line in sample
            ]
        )
        l1_norms.append(numpy.linalg.norm(corr_matrix, ord=1))
    return l1_norms


def get_inf_norm(tap_location_samples):
    inf_norms = []
    for sample in tap_location_samples:
        corr_matrix = numpy.array(
            [
                [float(log_line.split(",")[2]),
                 float(log_line.split(",")[3]),
                 float(log_line.split(",")[4])]
                for log_line in sample
            ]
        )
        inf_norms.append(numpy.linalg.norm(corr_matrix, ord=numpy.inf))
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
        corr_matrix = numpy.array(
            [
                [float(log_line.split(",")[2]),
                 float(log_line.split(",")[3]),
                 float(log_line.split(",")[4])]
                for log_line in sample
            ]
        )
        fro_norms.append(numpy.linalg.norm(corr_matrix, ord='fro'))
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


def featurize(samples):
    features = []
    for tap_location_sample in samples:
        means = get_sample_mean(tap_location_sample)
        std_dev = get_sample_std_dev(tap_location_sample)
        skew = get_sample_skew(tap_location_sample)
        kurtosis = get_sample_kurtosis(tap_location_sample)
        l1_norm = get_l1_norm(tap_location_sample)
        inf_norm = get_inf_norm(tap_location_sample)
        fro_norm = get_fro_norm(tap_location_sample)
        for i in range(len(means)):
            features.append(
                means[i] + std_dev[i] + skew[i] + kurtosis[i] +
                [l1_norm[i], inf_norm[i], fro_norm[i]]
            )
    return features


def featurize_new(samples):
    features = []
    for tap_location_sample in samples:
        p2p = get_sample_p2p(tap_location_sample)
        sign = get_peak_value_sign(tap_location_sample)
        rms = get_rms(tap_location_sample)
        for i in range(len(p2p)):
            features.append(
                p2p[i] + sign[i] + rms[i]
            )
    return features


def featurize_combined(samples):
    features = []
    for tap_location_sample in samples:
        means = get_sample_mean(tap_location_sample)
        std_dev = get_sample_std_dev(tap_location_sample)
        skew = get_sample_skew(tap_location_sample)
        kurtosis = get_sample_kurtosis(tap_location_sample)
        l1_norm = get_l1_norm(tap_location_sample)
        inf_norm = get_inf_norm(tap_location_sample)
        fro_norm = get_fro_norm(tap_location_sample)
        p2p = get_sample_p2p(tap_location_sample)
        sign = get_peak_value_sign(tap_location_sample)
        rms = get_rms(tap_location_sample)
        for i in range(len(means)):
            features.append(
                means[i] + std_dev[i] + skew[i] + kurtosis[i] +
                [l1_norm[i], inf_norm[i], fro_norm[i]] +
                p2p[i] + sign[i] + rms[i]
            )
    return features


def make_tap_occurrence_data(file_name):
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
                                                       RIGHT_HAND)
    rhand_p_samples_gyro = get_positive_tap_samples(GYROSCOPE, RIGHT_HAND)
    lhand_n_samples_lin_acc = get_negative_tap_samples(LINEAR_ACCELEROMETER,
                                                       LEFT_HAND)
    lhand_n_samples_gyro = get_negative_tap_samples(GYROSCOPE, LEFT_HAND)
    rhand_n_samples_lin_acc = get_negative_tap_samples(LINEAR_ACCELEROMETER,
                                                       RIGHT_HAND)
    rhand_n_samples_gyro = get_negative_tap_samples(GYROSCOPE, RIGHT_HAND)

    # First do positive linear accelerometer samples
    lin_acc_p_samples = lhand_p_samples_lin_acc + rhand_p_samples_lin_acc
    lin_acc_p_features = featurize(lin_acc_p_samples)
    # Add on positive gyroscope samples
    gyro_p_samples = lhand_p_samples_gyro + rhand_p_samples_gyro
    gyro_p_features = featurize(gyro_p_samples)

    # Synthesize positive tap features
    positive_features = []
    for i in range(len(lin_acc_p_features)):
        positive_features.append(lin_acc_p_features[i] + gyro_p_features[i])

    # Negative linear accelerometer samples
    lin_acc_n_samples = lhand_n_samples_lin_acc + rhand_n_samples_lin_acc
    lin_acc_n_features = featurize(lin_acc_n_samples)
    # Negative gyro samples
    gyro_n_samples = lhand_n_samples_gyro + rhand_n_samples_gyro
    gyro_n_features = featurize(gyro_n_samples)

    # Synthesize negative tap features
    negative_features = []
    for i in range(len(lin_acc_n_features)):
        negative_features.append(lin_acc_n_features[i] + gyro_n_features[i])

    # Now we write to the file.
    with open(
            "training/" + file_name + ".unscaled", 'w', encoding='utf-8') as file:
        for feature_vector in positive_features:
            file.write("+1 ")
            for i in range(len(feature_vector)):
                file.write(str(i + 1) + ":" + str(feature_vector[i]) + " ")
            file.write('\n')
        for feature_vector in negative_features:
            file.write("-1 ")
            for i in range(len(feature_vector)):
                file.write(str(i + 1) + ":" + str(feature_vector[i]) + " ")
            file.write('\n')


def make_tap_occurrence_data_new(file_name):
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
                                                       RIGHT_HAND)
    rhand_p_samples_gyro = get_positive_tap_samples(GYROSCOPE, RIGHT_HAND)
    lhand_n_samples_lin_acc = get_negative_tap_samples(LINEAR_ACCELEROMETER,
                                                       LEFT_HAND)
    lhand_n_samples_gyro = get_negative_tap_samples(GYROSCOPE, LEFT_HAND)
    rhand_n_samples_lin_acc = get_negative_tap_samples(LINEAR_ACCELEROMETER,
                                                       RIGHT_HAND)
    rhand_n_samples_gyro = get_negative_tap_samples(GYROSCOPE, RIGHT_HAND)

    # First do positive linear accelerometer samples
    lin_acc_p_samples = lhand_p_samples_lin_acc + rhand_p_samples_lin_acc
    lin_acc_p_features = featurize_new(lin_acc_p_samples)
    # Add on positive gyroscope samples
    gyro_p_samples = lhand_p_samples_gyro + rhand_p_samples_gyro
    gyro_p_features = featurize_new(gyro_p_samples)

    # Synthesize positive tap features
    positive_features = []
    for i in range(len(lin_acc_p_features)):
        positive_features.append(lin_acc_p_features[i] + gyro_p_features[i])

    # Negative linear accelerometer samples
    lin_acc_n_samples = lhand_n_samples_lin_acc + rhand_n_samples_lin_acc
    lin_acc_n_features = featurize_new(lin_acc_n_samples)
    # Negative gyro samples
    gyro_n_samples = lhand_n_samples_gyro + rhand_n_samples_gyro
    gyro_n_features = featurize_new(gyro_n_samples)

    # Synthesize negative tap features
    negative_features = []
    for i in range(len(lin_acc_n_features)):
        negative_features.append(lin_acc_n_features[i] + gyro_n_features[i])

    # Now we write to the file.
    with open(
            "training/" + file_name + ".unscaled", 'w', encoding='utf-8') as file:
        for feature_vector in positive_features:
            file.write("+1 ")
            for i in range(len(feature_vector)):
                file.write(str(i + 1) + ":" + str(feature_vector[i]) + " ")
            file.write('\n')
        for feature_vector in negative_features:
            file.write("-1 ")
            for i in range(len(feature_vector)):
                file.write(str(i + 1) + ":" + str(feature_vector[i]) + " ")
            file.write('\n')


def make_tap_occurrence_data_combined(file_name):
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
                                                       RIGHT_HAND)
    rhand_p_samples_gyro = get_positive_tap_samples(GYROSCOPE, RIGHT_HAND)
    lhand_n_samples_lin_acc = get_negative_tap_samples(LINEAR_ACCELEROMETER,
                                                       LEFT_HAND)
    lhand_n_samples_gyro = get_negative_tap_samples(GYROSCOPE, LEFT_HAND)
    rhand_n_samples_lin_acc = get_negative_tap_samples(LINEAR_ACCELEROMETER,
                                                       RIGHT_HAND)
    rhand_n_samples_gyro = get_negative_tap_samples(GYROSCOPE, RIGHT_HAND)

    # First do positive linear accelerometer samples
    lin_acc_p_samples = lhand_p_samples_lin_acc + rhand_p_samples_lin_acc
    lin_acc_p_features = featurize_combined(lin_acc_p_samples)
    # Add on positive gyroscope samples
    gyro_p_samples = lhand_p_samples_gyro + rhand_p_samples_gyro
    gyro_p_features = featurize_combined(gyro_p_samples)

    # Synthesize positive tap features
    positive_features = []
    for i in range(len(lin_acc_p_features)):
        positive_features.append(lin_acc_p_features[i] + gyro_p_features[i])

    # Negative linear accelerometer samples
    lin_acc_n_samples = lhand_n_samples_lin_acc + rhand_n_samples_lin_acc
    lin_acc_n_features = featurize_combined(lin_acc_n_samples)
    # Negative gyro samples
    gyro_n_samples = lhand_n_samples_gyro + rhand_n_samples_gyro
    gyro_n_features = featurize_combined(gyro_n_samples)

    # Synthesize negative tap features
    negative_features = []
    for i in range(len(lin_acc_n_features)):
        negative_features.append(lin_acc_n_features[i] + gyro_n_features[i])

    # Now we write to the file.
    with open(
            "training/" + file_name + ".unscaled", 'w', encoding='utf-8') as file:
        for feature_vector in positive_features:
            file.write("+1 ")
            for i in range(len(feature_vector)):
                file.write(str(i + 1) + ":" + str(feature_vector[i]) + " ")
            file.write('\n')
        for feature_vector in negative_features:
            file.write("-1 ")
            for i in range(len(feature_vector)):
                file.write(str(i + 1) + ":" + str(feature_vector[i]) + " ")
            file.write('\n')


def make_hand_data_2p(file_name):
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

    # Get angles
    lhand_angles = get_angle(lhand_left_taps) + get_angle(lhand_right_taps)
    rhand_angles = get_angle(rhand_left_taps) + get_angle(rhand_right_taps)

    # Write to training file
    with open("training/" + file_name + ".unscaled", 'w', encoding='utf-8') as \
            file:
        for angle_sample in lhand_angles:
            file.write("+1 1:" + str(angle_sample) + '\n')
        for angle_sample in rhand_angles:
            file.write("-1 1:" + str(angle_sample) + '\n')


def make_hand_data_5p(file_name):
    """
    Creates the left/right hand training/testing file in the format:
    <hand[-1 for left, +1 for right]> <index1>:<angle at max magnitude> ...
    :param file_name: Name of training file to be written
    :return:
    """
    # Get lists of data windows for left/right hand taps
    lhand_p_samples_lin_acc = get_positive_tap_samples(LINEAR_ACCELEROMETER,
                                                       LEFT_HAND)
    lhand_p_samples_gyro = get_positive_tap_samples(GYROSCOPE, LEFT_HAND)
    rhand_p_samples_lin_acc = get_positive_tap_samples(LINEAR_ACCELEROMETER,
                                                       RIGHT_HAND)
    rhand_p_samples_gyro = get_positive_tap_samples(GYROSCOPE, RIGHT_HAND)

    # Angles not calculated in featurize() since only lin acc is relevant
    lhand_angles = []
    for tap_location_sample in lhand_p_samples_lin_acc:
        lhand_angles = lhand_angles + get_angle(tap_location_sample)
    rhand_angles = []
    for tap_location_sample in rhand_p_samples_lin_acc:
        rhand_angles = rhand_angles + get_angle(tap_location_sample)

    # Other features
    lhand_lin_acc_features = featurize(lhand_p_samples_lin_acc)
    lhand_gyro_features = featurize(lhand_p_samples_gyro)
    rhand_lin_acc_features = featurize(rhand_p_samples_lin_acc)
    rhand_gyro_features = featurize(rhand_p_samples_gyro)

    lhand_features = []
    for i in range(len(lhand_lin_acc_features)):
        lhand_features.append([lhand_angles[i]] + lhand_lin_acc_features[i] +
                              lhand_gyro_features[i])

    rhand_features = []
    for i in range(len(rhand_lin_acc_features)):
        rhand_features.append([rhand_angles[i]] + rhand_lin_acc_features[i] +
                              rhand_gyro_features[i])

    # Now we write to the file.
    with open(
            "training/" + file_name + ".unscaled", 'w', encoding='utf-8') as file:
        for feature_vector in lhand_features:
            file.write("+1 ")
            for i in range(len(feature_vector)):
                file.write(str(i + 1) + ":" + str(feature_vector[i]) + " ")
            file.write('\n')
        for feature_vector in rhand_features:
            file.write("-1 ")
            for i in range(len(feature_vector)):
                file.write(str(i + 1) + ":" + str(feature_vector[i]) + " ")
            file.write('\n')


def make_hand_data_5p_no_angles(file_name):
    """
    Creates the left/right hand training/testing file in the format:
    <hand[-1 for left, +1 for right]> <index1>:<angle at max magnitude> ...
    :param file_name: Name of training file to be written
    :return:
    """
    # Get lists of data windows for left/right hand taps
    lhand_p_samples_lin_acc = get_positive_tap_samples(LINEAR_ACCELEROMETER,
                                                       LEFT_HAND)
    lhand_p_samples_gyro = get_positive_tap_samples(GYROSCOPE, LEFT_HAND)
    rhand_p_samples_lin_acc = get_positive_tap_samples(LINEAR_ACCELEROMETER,
                                                       RIGHT_HAND)
    rhand_p_samples_gyro = get_positive_tap_samples(GYROSCOPE, RIGHT_HAND)

    # Other features
    lhand_lin_acc_features = featurize(lhand_p_samples_lin_acc)
    lhand_gyro_features = featurize(lhand_p_samples_gyro)
    rhand_lin_acc_features = featurize(rhand_p_samples_lin_acc)
    rhand_gyro_features = featurize(rhand_p_samples_gyro)

    lhand_features = []
    for i in range(len(lhand_lin_acc_features)):
        lhand_features.append(lhand_lin_acc_features[i] +
                              lhand_gyro_features[i])

    rhand_features = []
    for i in range(len(rhand_lin_acc_features)):
        rhand_features.append(rhand_lin_acc_features[i] +
                              rhand_gyro_features[i])

    # Now we write to the file.
    with open(
            "training/" + file_name + ".unscaled", 'w', encoding='utf-8') as file:
        for feature_vector in lhand_features:
            file.write("+1 ")
            for i in range(len(feature_vector)):
                file.write(str(i + 1) + ":" + str(feature_vector[i]) + " ")
            file.write('\n')
        for feature_vector in rhand_features:
            file.write("-1 ")
            for i in range(len(feature_vector)):
                file.write(str(i + 1) + ":" + str(feature_vector[i]) + " ")
            file.write('\n')


def make_hand_data_5p_new(file_name):
    """
    Creates the left/right hand training/testing file in the format:
    <hand[-1 for left, +1 for right]> <index1>:<angle at max magnitude> ...
    :param file_name: Name of training file to be written
    :return:
    """
    # Get lists of data windows for left/right hand taps
    lhand_p_samples_lin_acc = get_positive_tap_samples(LINEAR_ACCELEROMETER,
                                                       LEFT_HAND)
    lhand_p_samples_gyro = get_positive_tap_samples(GYROSCOPE, LEFT_HAND)
    rhand_p_samples_lin_acc = get_positive_tap_samples(LINEAR_ACCELEROMETER,
                                                       RIGHT_HAND)
    rhand_p_samples_gyro = get_positive_tap_samples(GYROSCOPE, RIGHT_HAND)

    # Angles not calculated in featurize() since only lin acc is relevant
    lhand_angles = []
    for tap_location_sample in lhand_p_samples_lin_acc:
        lhand_angles = lhand_angles + get_angle(tap_location_sample)
    rhand_angles = []
    for tap_location_sample in rhand_p_samples_lin_acc:
        rhand_angles = rhand_angles + get_angle(tap_location_sample)

    # Other features
    lhand_lin_acc_features = featurize_new(lhand_p_samples_lin_acc)
    lhand_gyro_features = featurize_new(lhand_p_samples_gyro)
    rhand_lin_acc_features = featurize_new(rhand_p_samples_lin_acc)
    rhand_gyro_features = featurize_new(rhand_p_samples_gyro)

    lhand_features = []
    for i in range(len(lhand_lin_acc_features)):
        lhand_features.append([lhand_angles[i]] + lhand_lin_acc_features[i] +
                              lhand_gyro_features[i])

    rhand_features = []
    for i in range(len(rhand_lin_acc_features)):
        rhand_features.append([rhand_angles[i]] + rhand_lin_acc_features[i] +
                              rhand_gyro_features[i])

    # Now we write to the file.
    with open(
            "training/" + file_name + ".unscaled", 'w', encoding='utf-8') as file:
        for feature_vector in lhand_features:
            file.write("+1 ")
            for i in range(len(feature_vector)):
                file.write(str(i + 1) + ":" + str(feature_vector[i]) + " ")
            file.write('\n')
        for feature_vector in rhand_features:
            file.write("-1 ")
            for i in range(len(feature_vector)):
                file.write(str(i + 1) + ":" + str(feature_vector[i]) + " ")
            file.write('\n')


def make_hand_data_5p_new_no_angles(file_name):
    """
    Creates the left/right hand training/testing file in the format:
    <hand[-1 for left, +1 for right]> <index1>:<angle at max magnitude> ...
    :param file_name: Name of training file to be written
    :return:
    """
    # Get lists of data windows for left/right hand taps
    lhand_p_samples_lin_acc = get_positive_tap_samples(LINEAR_ACCELEROMETER,
                                                       LEFT_HAND)
    lhand_p_samples_gyro = get_positive_tap_samples(GYROSCOPE, LEFT_HAND)
    rhand_p_samples_lin_acc = get_positive_tap_samples(LINEAR_ACCELEROMETER,
                                                       RIGHT_HAND)
    rhand_p_samples_gyro = get_positive_tap_samples(GYROSCOPE, RIGHT_HAND)

    # Features
    lhand_lin_acc_features = featurize_new(lhand_p_samples_lin_acc)
    lhand_gyro_features = featurize_new(lhand_p_samples_gyro)
    rhand_lin_acc_features = featurize_new(rhand_p_samples_lin_acc)
    rhand_gyro_features = featurize_new(rhand_p_samples_gyro)

    lhand_features = []
    for i in range(len(lhand_lin_acc_features)):
        lhand_features.append(lhand_lin_acc_features[i] +
                              lhand_gyro_features[i])

    rhand_features = []
    for i in range(len(rhand_lin_acc_features)):
        rhand_features.append(rhand_lin_acc_features[i] +
                              rhand_gyro_features[i])

    # Now we write to the file.
    with open(
            "training/" + file_name + ".unscaled", 'w', encoding='utf-8') as file:
        for feature_vector in lhand_features:
            file.write("+1 ")
            for i in range(len(feature_vector)):
                file.write(str(i + 1) + ":" + str(feature_vector[i]) + " ")
            file.write('\n')
        for feature_vector in rhand_features:
            file.write("-1 ")
            for i in range(len(feature_vector)):
                file.write(str(i + 1) + ":" + str(feature_vector[i]) + " ")
            file.write('\n')


def make_hand_data_5p_angle(file_name):
    # Get lists of data windows for left/right hand taps
    lhand_p_samples_lin_acc = get_positive_tap_samples(LINEAR_ACCELEROMETER,
                                                       LEFT_HAND)
    rhand_p_samples_lin_acc = get_positive_tap_samples(LINEAR_ACCELEROMETER,
                                                       RIGHT_HAND)

    # Get angles from linear accelerometer
    lhand_angles = []
    for tap_location_sample in lhand_p_samples_lin_acc:
        lhand_angles = lhand_angles + get_angle(tap_location_sample)
    rhand_angles = []
    for tap_location_sample in rhand_p_samples_lin_acc:
        rhand_angles = rhand_angles + get_angle(tap_location_sample)

    # Write to training file
    with open("training/" + file_name + ".unscaled", 'w', encoding='utf-8') as \
            file:
        for angle_sample in lhand_angles:
            file.write("+1 1:" + str(angle_sample) + '\n')
        for angle_sample in rhand_angles:
            file.write("-1 1:" + str(angle_sample) + '\n')


def make_hand_data_5p_combined(file_name):
    """
    Creates the left/right hand training/testing file in the format:
    <hand[-1 for left, +1 for right]> <index1>:<angle at max magnitude> ...
    :param file_name: Name of training file to be written
    :return:
    """
    # Get lists of data windows for left/right hand taps
    lhand_p_samples_lin_acc = get_positive_tap_samples(LINEAR_ACCELEROMETER,
                                                       LEFT_HAND)
    lhand_p_samples_gyro = get_positive_tap_samples(GYROSCOPE, LEFT_HAND)
    rhand_p_samples_lin_acc = get_positive_tap_samples(LINEAR_ACCELEROMETER,
                                                       RIGHT_HAND)
    rhand_p_samples_gyro = get_positive_tap_samples(GYROSCOPE, RIGHT_HAND)

    # Angles not calculated in featurize() since only lin acc is relevant
    lhand_angles = []
    for tap_location_sample in lhand_p_samples_lin_acc:
        lhand_angles = lhand_angles + get_angle(tap_location_sample)
    rhand_angles = []
    for tap_location_sample in rhand_p_samples_lin_acc:
        rhand_angles = rhand_angles + get_angle(tap_location_sample)

    # Other features
    lhand_lin_acc_features = featurize_combined(lhand_p_samples_lin_acc)
    lhand_gyro_features = featurize_combined(lhand_p_samples_gyro)
    rhand_lin_acc_features = featurize_combined(rhand_p_samples_lin_acc)
    rhand_gyro_features = featurize_combined(rhand_p_samples_gyro)

    lhand_features = []
    for i in range(len(lhand_lin_acc_features)):
        lhand_features.append([lhand_angles[i]] + lhand_lin_acc_features[i] +
                              lhand_gyro_features[i])

    rhand_features = []
    for i in range(len(rhand_lin_acc_features)):
        rhand_features.append([rhand_angles[i]] + rhand_lin_acc_features[i] +
                              rhand_gyro_features[i])

    # Now we write to the file.
    with open(
            "training/" + file_name + ".unscaled", 'w', encoding='utf-8') as file:
        for feature_vector in lhand_features:
            file.write("+1 ")
            for i in range(len(feature_vector)):
                file.write(str(i + 1) + ":" + str(feature_vector[i]) + " ")
            file.write('\n')
        for feature_vector in rhand_features:
            file.write("-1 ")
            for i in range(len(feature_vector)):
                file.write(str(i + 1) + ":" + str(feature_vector[i]) + " ")
            file.write('\n')


def make_left_hand_location_data(file_name):
    """
    Creates the data file for classifying tap position with the left hand in the
    format:
    <[tap location 1,...,5] <index1>:<x axis lin acc mean> ...
    :param file_name: Name of training file to be written
    :return:
    """
    # Get positive and negative samples
    lhand_p_samples_lin_acc = get_positive_tap_samples(LINEAR_ACCELEROMETER,
                                                       LEFT_HAND)
    lhand_p_samples_gyro = get_positive_tap_samples(GYROSCOPE, LEFT_HAND)

    # First featurize positive samples for the left hand
    lin_acc_p_features = featurize(lhand_p_samples_lin_acc)
    gyro_p_features = featurize(lhand_p_samples_gyro)

    # Synthesize positive tap features
    positive_features = []
    for i in range(len(lin_acc_p_features)):
        positive_features.append(lin_acc_p_features[i] + gyro_p_features[i])

    # Now we write to the file.
    with open(
            "training/" + file_name + ".unscaled", 'w', encoding='utf-8') as file:
        location = 1
        count = 1
        feature_count = len(positive_features)
        count_breakpoints = [int(feature_count * i / 5) for i in range(1, 6)]
        for feature_vector in positive_features:
            file.write(str(location) + " ")
            for i in range(len(feature_vector)):
                file.write(str(i + 1) + ":" + str(feature_vector[i]) + " ")
            file.write('\n')
            if count in count_breakpoints:
                location += 1
            count += 1


def make_left_hand_location_data_new(file_name):
    """
    Creates the data file for classifying tap position with the left hand in the
    format:
    <[tap location 1,...,5] <index1>:<x axis lin acc mean> ...
    :param file_name: Name of training file to be written
    :return:
    """
    # Get positive and negative samples
    lhand_p_samples_lin_acc = get_positive_tap_samples(LINEAR_ACCELEROMETER,
                                                       LEFT_HAND)
    lhand_p_samples_gyro = get_positive_tap_samples(GYROSCOPE, LEFT_HAND)

    # First featurize positive samples for the left hand
    lin_acc_p_features = featurize_new(lhand_p_samples_lin_acc)
    gyro_p_features = featurize_new(lhand_p_samples_gyro)

    # Synthesize positive tap features
    positive_features = []
    for i in range(len(lin_acc_p_features)):
        positive_features.append(lin_acc_p_features[i] + gyro_p_features[i])

    # Now we write to the file.
    with open(
            "training/" + file_name + ".unscaled", 'w', encoding='utf-8') as file:
        location = 1
        count = 1
        feature_count = len(positive_features)
        count_breakpoints = [int(feature_count * i / 5) for i in range(1, 6)]
        for feature_vector in positive_features:
            file.write(str(location) + " ")
            for i in range(len(feature_vector)):
                file.write(str(i + 1) + ":" + str(feature_vector[i]) + " ")
            file.write('\n')
            if count in count_breakpoints:
                location += 1
            count += 1


def make_left_hand_location_data_combined(file_name):
    """
    Creates the data file for classifying tap position with the left hand in the
    format:
    <[tap location 1,...,5] <index1>:<x axis lin acc mean> ...
    :param file_name: Name of training file to be written
    :return:
    """
    # Get positive and negative samples
    lhand_p_samples_lin_acc = get_positive_tap_samples(LINEAR_ACCELEROMETER,
                                                       LEFT_HAND)
    lhand_p_samples_gyro = get_positive_tap_samples(GYROSCOPE, LEFT_HAND)

    # First featurize positive samples for the left hand
    lin_acc_p_features = featurize_combined(lhand_p_samples_lin_acc)
    gyro_p_features = featurize_combined(lhand_p_samples_gyro)

    # Synthesize positive tap features
    positive_features = []
    for i in range(len(lin_acc_p_features)):
        positive_features.append(lin_acc_p_features[i] + gyro_p_features[i])

    # Now we write to the file.
    with open(
            "training/" + file_name + ".unscaled", 'w', encoding='utf-8') as file:
        location = 1
        count = 1
        feature_count = len(positive_features)
        count_breakpoints = [int(feature_count * i / 5) for i in range(1, 6)]
        for feature_vector in positive_features:
            file.write(str(location) + " ")
            for i in range(len(feature_vector)):
                file.write(str(i + 1) + ":" + str(feature_vector[i]) + " ")
            file.write('\n')
            if count in count_breakpoints:
                location += 1
            count += 1


def make_right_hand_location_data(file_name):
    """
    Creates the data file for classifying tap position with the left hand in the
    format:
    <[tap location 1,...,5] <index1>:<x axis lin acc mean> ...
    :param file_name: Name of training file to be written
    :return:
    """
    # Get positive and negative samples
    rhand_p_samples_lin_acc = get_positive_tap_samples(LINEAR_ACCELEROMETER,
                                                       RIGHT_HAND)
    rhand_p_samples_gyro = get_positive_tap_samples(GYROSCOPE, RIGHT_HAND)

    # First featurize positive samples for the right hand
    lin_acc_p_features = featurize(rhand_p_samples_lin_acc)
    gyro_p_features = featurize(rhand_p_samples_gyro)

    # Synthesize positive tap features
    positive_features = []
    for i in range(len(lin_acc_p_features)):
        positive_features.append(lin_acc_p_features[i] + gyro_p_features[i])

    # Now we write to the file.
    with open(
            "training/" + file_name + ".unscaled", 'w', encoding='utf-8') as file:
        location = 1
        count = 1
        feature_count = len(positive_features)
        count_breakpoints = [int(feature_count * i / 5) for i in range(1, 6)]
        for feature_vector in positive_features:
            file.write(str(location) + " ")
            for i in range(len(feature_vector)):
                file.write(str(i + 1) + ":" + str(feature_vector[i]) + " ")
            file.write('\n')
            if count in count_breakpoints:
                location += 1
            count += 1


def make_right_hand_location_data_new(file_name):
    """
    Creates the data file for classifying tap position with the left hand in the
    format:
    <[tap location 1,...,5] <index1>:<x axis lin acc mean> ...
    :param file_name: Name of training file to be written
    :return:
    """
    # Get positive and negative samples
    rhand_p_samples_lin_acc = get_positive_tap_samples(LINEAR_ACCELEROMETER,
                                                       RIGHT_HAND)
    rhand_p_samples_gyro = get_positive_tap_samples(GYROSCOPE, RIGHT_HAND)

    # First featurize positive samples for the right hand
    lin_acc_p_features = featurize_new(rhand_p_samples_lin_acc)
    gyro_p_features = featurize_new(rhand_p_samples_gyro)

    # Synthesize positive tap features
    positive_features = []
    for i in range(len(lin_acc_p_features)):
        positive_features.append(lin_acc_p_features[i] + gyro_p_features[i])

    # Now we write to the file.
    with open(
            "training/" + file_name + ".unscaled", 'w', encoding='utf-8') as file:
        location = 1
        count = 1
        feature_count = len(positive_features)
        count_breakpoints = [int(feature_count * i / 5) for i in range(1, 6)]
        for feature_vector in positive_features:
            file.write(str(location) + " ")
            for i in range(len(feature_vector)):
                file.write(str(i + 1) + ":" + str(feature_vector[i]) + " ")
            file.write('\n')
            if count in count_breakpoints:
                location += 1
            count += 1


def make_right_hand_location_data_combined(file_name):
    """
    Creates the data file for classifying tap position with the left hand in the
    format:
    <[tap location 1,...,5] <index1>:<x axis lin acc mean> ...
    :param file_name: Name of training file to be written
    :return:
    """
    # Get positive and negative samples
    rhand_p_samples_lin_acc = get_positive_tap_samples(LINEAR_ACCELEROMETER,
                                                       RIGHT_HAND)
    rhand_p_samples_gyro = get_positive_tap_samples(GYROSCOPE, RIGHT_HAND)

    # First featurize positive samples for the right hand
    lin_acc_p_features = featurize_combined(rhand_p_samples_lin_acc)
    gyro_p_features = featurize_combined(rhand_p_samples_gyro)

    # Synthesize positive tap features
    positive_features = []
    for i in range(len(lin_acc_p_features)):
        positive_features.append(lin_acc_p_features[i] + gyro_p_features[i])

    # Now we write to the file.
    with open(
            "training/" + file_name + ".unscaled", 'w', encoding='utf-8') as file:
        location = 1
        count = 1
        feature_count = len(positive_features)
        count_breakpoints = [int(feature_count * i / 5) for i in range(1, 6)]
        for feature_vector in positive_features:
            file.write(str(location) + " ")
            for i in range(len(feature_vector)):
                file.write(str(i + 1) + ":" + str(feature_vector[i]) + " ")
            file.write('\n')
            if count in count_breakpoints:
                location += 1
            count += 1


# Program starts here ----------------------------------------------------------

clean_logs()

make_tap_occurrence_data("tap_occurrence")
# make_tap_occurrence_data_new("tap_occurrence_new_features")
# make_tap_occurrence_data_combined("tap_occurrence_combined")

# make_hand_data_2p("hand_2p")
make_hand_data_5p("hand_5p")
# make_hand_data_5p_no_angles("hand_5p_no_angles")
# make_hand_data_5p_new("hand_5p_new_features")
# make_hand_data_5p_new_no_angles("hand_5p_new_no_angles")
# make_hand_data_5p_angle("hand_5p_angles_only")
# make_hand_data_5p_combined("hand_5p_combined")

make_left_hand_location_data("location_lhand")
# make_left_hand_location_data_new("location_lhand_new_features")
# make_left_hand_location_data_combined("location_lhand_combined")

make_right_hand_location_data("location_rhand")
# make_right_hand_location_data_new("location_rhand_new_features")
# make_right_hand_location_data_combined("location_rhand_combined")
