import json
import os


def process_2p_left_hand_lin_acc():
    log_dir = os.path.join('logs', '2_points', 'left_hand', 'lin_acc')
    file_names = os.listdir(log_dir)  # list of strings, '1', '2', etc
    for file_name in file_names:
        file = open(os.path.join(log_dir, file_name), 'r+', encoding='utf-8')
        content = file.read()
        file.close()

        split_lines = content.splitlines()

        left_taps = []
        right_taps = []
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

        print(json.dumps(left_taps, indent=2))
        print(json.dumps(right_taps, indent=2))
        # print(right_taps)


process_2p_left_hand_lin_acc()

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
