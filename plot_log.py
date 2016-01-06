import os
import datetime
from matplotlib import pyplot


log_dir = os.path.join('logs', '5_points', 'left_hand', 'lin_acc')
file_names = os.listdir(log_dir)  # list of strings, '1', '2', etc

with open(os.path.join(log_dir, file_names[4])) as file:
    content = file.read()
    split_lines = content.splitlines()
    x_acc_values = []
    timestamps = []
    readable_timestamps = []
    for i in range(102):
        if int(split_lines[i].split(",")[1]) == 1:
            x_acc_values.append(float(split_lines[i].split(",")[2]))
            timestamps.append(int(split_lines[i].split(",")[0]))
    print("Max acceleration: " + str(max(x_acc_values, key=abs)))
    [readable_timestamps.append(datetime.timedelta(milliseconds=timestamp)) for
     timestamp in timestamps]
    pyplot.plot(timestamps, x_acc_values, 'o')
    pyplot.show()
