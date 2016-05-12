from training_functions import get_sample_mean
from training_functions import get_sample_std_dev
from training_functions import get_sample_skew
from training_functions import get_sample_kurtosis
from training_functions import get_l1_norm
from training_functions import get_inf_norm
from training_functions import get_fro_norm
from training_functions import get_angle
from training_functions import featurize

from scipy import stats

lin_acc_sample = [[
    "1454432615847,1,0.6199161410331726,-1.840886116027832,4.363898277282715",
    "1454432615867,1,-1.295964002609253,1.3011603355407715,1.976028323173523",
    "1454432615885,1,-1.0550503730773926,1.2519176006317139,-1.0213059186935425",
    "1454432615905,1,1.2163758277893066,0.7365642189979553,-2.471317768096924",
    "1454432615928,1,1.695397973060608,0.42944473028182983,-2.4582371711730957",
    "1454432615949,1,0.42726776003837585,0.16897183656692505,-1.2394062280654907",
    "1454432615966,1,-0.2335493564605713,0.31944385170936584,0.7737187743186951",
    "1454432615984,1,-0.1027030199766159,0.18706630170345306,0.9212544560432434",
    "1454432616005,1,0.18074658513069153,-0.13008154928684235,0.5268833637237549",
    "1454432616024,1,0.48971420526504517,-0.3674953579902649,0.07739304006099701",
    "1454432616046,1,0.749410092830658,-0.40540364384651184,0.06440963596105576",
    "1454432616067,1,0.6956585049629211,-0.32503655552864075,0.20324644446372986",
    "1454432616085,1,0.7970908880233765,-0.22689619660377502,0.2895505130290985",
    "1454432616105,1,0.8320264220237732,-0.014623276889324188,0.4061398506164551",
    "1454432616128,1,0.7162103056907654,0.023281799629330635,0.6019318699836731",
]]

gyro_sample = [[
    "1454432615840, 1, -0.9426119923591614, -0.6642239689826965, -0.5274720191955566",
    "1454432615860, 1, 0.7069590091705322, -0.2857140004634857, 0.15995100140571594",
    "1454432615880, 1, 1.1025630235671997, -0.26129400730133057, 0.5873010158538818",
    "1454432615899, 1, 0.7142850160598755, -0.16361400485038757, 0.6336989998817444",
    "1454432615920, 1, 0.07936500012874603, -0.1892549991607666, 0.31013399362564087",
    "1454432615940, 1, -0.4175820052623749, -0.12209999561309814, 0.06227099895477295",
    "1454432615960, 1, -0.3565320074558258, 0.15995100140571594, 0.167276993393898",
    "1454432615980, 1, -0.07692299783229828, 0.2332109957933426, 0.2808299958705902",
    "1454432616001, 1, 0.1562879979610443, 0.2649570107460022, 0.34432199597358704",
    "1454432616021, 1, 0.2527469992637634, 0.28815600275993347, 0.33821699023246765",
    "1454432616042, 1, 0.22100099921226501, 0.2869350016117096, 0.2747249901294708",
    "1454432616062, 1, 0.14285700023174286, 0.13064700365066528, 0.15018299221992493",
    "1454432616081, 1, -0.050060998648405075, 0.020756999030709267, 0.11843699961900711",
    "1454432616102, 1, -0.11233200132846832, -0.040293000638484955, 0.08546999841928482",
    "1454432616122, 1, -0.12576299905776978, -0.07203900068998337, 0.08546999841928482",
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


def test_get_pearsonr(lin_acc_sample, gyro_sample):
    lin_acc_container = []
    for sample in lin_acc_sample:
        lin_acc_x = [float(log_line.split(",")[2]) for log_line in sample]
        lin_acc_y = [float(log_line.split(",")[3]) for log_line in sample]
        lin_acc_z = [float(log_line.split(",")[4]) for log_line in sample]
        lin_acc_container.append([lin_acc_x, lin_acc_y, lin_acc_z])
    gyro_container = []
    for sample in gyro_sample:
        gyro_x = [float(log_line.split(",")[2]) for log_line in sample]
        gyro_y = [float(log_line.split(",")[3]) for log_line in sample]
        gyro_z = [float(log_line.split(",")[4]) for log_line in sample]
        gyro_container.append([gyro_x, gyro_y, gyro_z])

    pearson_coeff = []
    for i in range(len(lin_acc_container)):  # Two containers have same len
        sample_p_coeff = []
        for j in range(len(lin_acc_container[0])):  # 0-2
            # lin acc x to gyro x/y/z
            sample_p_coeff.append(
                stats.pearsonr(lin_acc_container[i][0],
                               gyro_container[i][j])[0]
            )
        for j in range(len(lin_acc_container[0])):
            sample_p_coeff.append(
                stats.pearsonr(lin_acc_container[i][1],
                               gyro_container[i][j])[0]
            )
        for j in range(len(lin_acc_container[0])):
            sample_p_coeff.append(
                stats.pearsonr(lin_acc_container[i][2],
                               gyro_container[i][j])[0]
            )
        pearson_coeff.append(sample_p_coeff)
    return pearson_coeff

print(get_sample_mean(lin_acc_sample))
print(get_sample_std_dev(lin_acc_sample))
print(get_sample_skew(lin_acc_sample))
print(get_sample_kurtosis(lin_acc_sample))
print(get_l1_norm(lin_acc_sample))
print(get_inf_norm(lin_acc_sample))
print(get_fro_norm(lin_acc_sample))
print(test_get_pearsonr(lin_acc_sample, gyro_sample))
# print(get_angle(lin_acc_sample))

# print(featurize([lin_acc_sample]))

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

