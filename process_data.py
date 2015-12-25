def process_left_hand():
    file = open('logs/2_points/left_hand/lin_acc/5', 'r+', encoding='utf-8')
    content = file.read()
    file.close()

    split_lines = content.splitlines()

    left_taps = []
    right_taps = []

    for i in range(len(split_lines)):
        left_tap_index = 0
        right_tap_index = 0
        if i > 1:  # Start at the second line
            split_line = split_lines[i].split(",")
            split_line_prev = split_lines[i-1].split(",")
            if int(split_line[1]) == 1 and int(split_line_prev[1]) == 1:
                left_taps[left_tap_index].append(split_lines[i])
            elif int(split_line[1]) == 1 and int(split_line_prev[1]) == 2:
                left_taps[left_tap_index].append(split_lines[i])
            elif int(split_line[1]) == 2 and int(split_line_prev[1]) == 1:
                left_tap_index += 1
                right_taps[right_tap_index].append(split_lines[i])
            elif int(split_line[1]) == 2 and int(split_line_prev[1]) == 2:
                right_taps[right_tap_index].append(split_lines[i])

        print(left_taps)


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
