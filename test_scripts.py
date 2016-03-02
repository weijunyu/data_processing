import os

from matplotlib import pyplot
from scipy import stats




# x = [1,3,5,7]
# y = [17,13,19,1]

# print(stats.pearsonr(x,y))


# pyplot.plot([1,2,3],[1,4,9], 'o')  # Vectors are x,y
# pyplot.axis([-1,10,-1,20])  # Vector is xmin, xmax, ymin, ymax
# pyplot.ylabel("Some numbers")
# pyplot.show()

# log_dir = os.path.join('logs', '2_points', 'left_hand', 'lin_acc')
# file_names = os.listdir(log_dir)  # list of strings, '1', '2', etc
#
# with open(os.path.join(log_dir, file_names[2])) as file:
#     content = file.read()
#     split_lines = content.splitlines()
#     x_acc_values = []
#     timestamps = []
#     for i in range(102):
#         if int(split_lines[i].split(",")[1]) == 1:
#             x_acc_values.append(float(split_lines[i].split(",")[2]))
#             timestamps.append(int(split_lines[i].split(",")[0]))
#     pyplot.plot(timestamps, x_acc_values, 'o')
#     pyplot.show()

