import os


def process_5p_left_hand_lin_acc():
    """
    Sorts data logged for the left hand, linear accelerometer, for 5 tap points
    :return: 5 lists containing data windows for each of 5 tap locations.
    """
    log_dir = os.path.join('logs', '5_points', 'left_hand', 'lin_acc')
    file_names = os.listdir(log_dir)  # list of strings, '1', '2', etc

    location_1 = []
    location_2 = []
    location_3 = []
    location_4 = []
    location_5 = []

    for file_name in file_names:
        file = open(os.path.join(log_dir, file_name), encoding='utf-8')
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
                    location_1.append(data_window.copy())
                    data_window.clear()
                    data_window.append(split_lines[i])
                elif int(split_line[1]) == 3 and int(split_line_prev[1]) == 2:
                    location_2.append(data_window.copy())
                    data_window.clear()
                    data_window.append(split_lines[i])
                elif int(split_line[1]) == 4 and int(split_line_prev[1]) == 3:
                    location_3.append(data_window.copy())
                    data_window.clear()
                    data_window.append(split_lines[i])
                elif int(split_line[1]) == 5 and int(split_line_prev[1]) == 4:
                    location_4.append(data_window.copy())
                    data_window.clear()
                    data_window.append(split_lines[i])
                elif int(split_line[1]) == 1 and int(split_line_prev[1]) == 5:
                    location_5.append(data_window.copy())
                    data_window.clear()
                    data_window.append(split_lines[i])
                if i == (len(split_lines) - 1):  # Last line
                    location_5.append(data_window.copy())

    return [location_1, location_2, location_3, location_4, location_5]


def process_5p_right_hand_lin_acc():
    """
    Sorts data logged for the right hand, linear accelerometer, for 5 tap points
    :return: 5 lists containing data windows for each of 5 tap locations.
    """
    log_dir = os.path.join('logs', '5_points', 'right_hand', 'lin_acc')
    file_names = os.listdir(log_dir)  # list of strings, '1', '2', etc

    location_1 = []
    location_2 = []
    location_3 = []
    location_4 = []
    location_5 = []

    for file_name in file_names:
        file = open(os.path.join(log_dir, file_name), encoding='utf-8')
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
                    location_1.append(data_window.copy())
                    data_window.clear()
                    data_window.append(split_lines[i])
                elif int(split_line[1]) == 3 and int(split_line_prev[1]) == 2:
                    location_2.append(data_window.copy())
                    data_window.clear()
                    data_window.append(split_lines[i])
                elif int(split_line[1]) == 4 and int(split_line_prev[1]) == 3:
                    location_3.append(data_window.copy())
                    data_window.clear()
                    data_window.append(split_lines[i])
                elif int(split_line[1]) == 5 and int(split_line_prev[1]) == 4:
                    location_4.append(data_window.copy())
                    data_window.clear()
                    data_window.append(split_lines[i])
                elif int(split_line[1]) == 1 and int(split_line_prev[1]) == 5:
                    location_5.append(data_window.copy())
                    data_window.clear()
                    data_window.append(split_lines[i])
                if i == (len(split_lines) - 1):  # Last line
                    location_5.append(data_window.copy())

    return [location_1, location_2, location_3, location_4, location_5]


def process_5p_left_hand_gyroscope():
    """
    Sorts data logged for the left hand, gyroscope, for 5 tap points
    :return: 5 lists containing data windows for each of 5 tap locations.
    """
    log_dir = os.path.join('logs', '5_points', 'left_hand', 'gyro')
    file_names = os.listdir(log_dir)  # list of strings, '1', '2', etc

    location_1 = []
    location_2 = []
    location_3 = []
    location_4 = []
    location_5 = []

    for file_name in file_names:
        file = open(os.path.join(log_dir, file_name), encoding='utf-8')
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
                    location_1.append(data_window.copy())
                    data_window.clear()
                    data_window.append(split_lines[i])
                elif int(split_line[1]) == 3 and int(split_line_prev[1]) == 2:
                    location_2.append(data_window.copy())
                    data_window.clear()
                    data_window.append(split_lines[i])
                elif int(split_line[1]) == 4 and int(split_line_prev[1]) == 3:
                    location_3.append(data_window.copy())
                    data_window.clear()
                    data_window.append(split_lines[i])
                elif int(split_line[1]) == 5 and int(split_line_prev[1]) == 4:
                    location_4.append(data_window.copy())
                    data_window.clear()
                    data_window.append(split_lines[i])
                elif int(split_line[1]) == 1 and int(split_line_prev[1]) == 5:
                    location_5.append(data_window.copy())
                    data_window.clear()
                    data_window.append(split_lines[i])
                if i == (len(split_lines) - 1):  # Last line
                    location_5.append(data_window.copy())

    return [location_1, location_2, location_3, location_4, location_5]


def process_5p_right_hand_gyroscope():
    """
    Sorts data logged for the right hand, linear accelerometer, for 5 tap points
    :return: 5 lists containing data windows for each of 5 tap locations.
    """
    log_dir = os.path.join('logs', '5_points', 'right_hand', 'gyroscope')
    file_names = os.listdir(log_dir)  # list of strings, '1', '2', etc

    location_1 = []
    location_2 = []
    location_3 = []
    location_4 = []
    location_5 = []

    for file_name in file_names:
        file = open(os.path.join(log_dir, file_name), encoding='utf-8')
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
                    location_1.append(data_window.copy())
                    data_window.clear()
                    data_window.append(split_lines[i])
                elif int(split_line[1]) == 3 and int(split_line_prev[1]) == 2:
                    location_2.append(data_window.copy())
                    data_window.clear()
                    data_window.append(split_lines[i])
                elif int(split_line[1]) == 4 and int(split_line_prev[1]) == 3:
                    location_3.append(data_window.copy())
                    data_window.clear()
                    data_window.append(split_lines[i])
                elif int(split_line[1]) == 5 and int(split_line_prev[1]) == 4:
                    location_4.append(data_window.copy())
                    data_window.clear()
                    data_window.append(split_lines[i])
                elif int(split_line[1]) == 1 and int(split_line_prev[1]) == 5:
                    location_5.append(data_window.copy())
                    data_window.clear()
                    data_window.append(split_lines[i])
                if i == (len(split_lines) - 1):  # Last line
                    location_5.append(data_window.copy())

    return [location_1, location_2, location_3, location_4, location_5]


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
        file = open(os.path.join(log_dir, file_name), encoding='utf-8')
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
        file = open(os.path.join(log_dir, file_name), encoding='utf-8')
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
                    right_taps.append(data_window.copy())
                    data_window.clear()
                    data_window.append(split_lines[i])
                elif int(split_line[1]) == 1 and int(split_line_prev[1]) == 2:
                    left_taps.append(data_window.copy())
                    data_window.clear()
                    data_window.append(split_lines[i])
                if i == (len(split_lines) - 1):  # Last line
                    left_taps.append(data_window.copy())

    return [left_taps, right_taps]
