from make_training_file import get_sample_mean
from make_training_file import get_sample_kurtosis
from make_training_file import get_sample_skew

lin_acc_sample = [[
    "1454432615403, 1, -0.5762239694595337, 0.16243454813957214,0.0730307474732399",
    "1454432615423, 1, -0.5722370743751526, 0.15929098427295685,0.049130234867334366",
    "1454432615445, 1, -0.5722819566726685, 0.16261398792266846,0.3850666582584381",
]]

gyro_sample = [[
    "1454432615398, 1, 0.021978000178933144, 0.23076899349689484,-0.41391900181770325",
    "1454432615419, 1, 0.020756999030709267, 0.27838799357414246,-0.40903499722480774",
    "1454432615439, 1, 0.07203900068998337, 0.3638579845428467, -0.40659299492836"
]]

# Things to test for tap detection:
# 1. mean
# 2. standard dev
# 3. skewness
# 4. kurtosis
# 5. l1 norm
# 6. infinite norm
# 7. frobenius norm
# 8. pearson coefficients

print(get_sample_mean(lin_acc_sample))
print(get_sample_skew(lin_acc_sample))

# import os
#
# from matplotlib import pyplot
# from scipy import stats

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

