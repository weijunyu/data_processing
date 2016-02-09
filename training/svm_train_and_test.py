import subprocess
import pprint
import numpy
import os


def cross_validate(parameters, file):
    c_low = parameters[0]
    c_high = parameters[1]
    gamma_low = parameters[2]
    gamma_high = parameters[3]
    results = []
    for c_power in numpy.linspace(c_low, c_high, 5):
        for gamma_power in numpy.linspace(gamma_low, gamma_high, 5):
            cost = 2 ** c_power
            gamma = 2 ** gamma_power
            training_process = subprocess.run(
                    ['svm-train.exe',
                     '-v', '10',  # 10 fold cross validation
                     '-c', str(cost),  # Cost
                     '-g', str(gamma),  # Gamma in kernel function
                     file],
                    stdout=subprocess.PIPE
            )

            result_string = str(
                    training_process.stdout.decode('utf-8')
            ).splitlines()
            percentage = float(result_string[-1].split(" ")[-1][:-1])
            results.append([c_power, gamma_power, percentage])
    return results


for file in os.listdir():
    file_name, file_ext = os.path.splitext(file)
    if file_ext == ".scaled":
        pprint.pprint(file)

        step = 18
        lower_bound = -6
        upper_bound = lower_bound + step  # initial bounds are -6 to 12
        params = (lower_bound, upper_bound,
                  lower_bound, upper_bound)
        for i in range(4):
            results = cross_validate(params, file)
            percentage_list = [result[2] for result in results]
            max_percentage = max(percentage_list)
            max_percentage_index = percentage_list.index(max_percentage)
            optimal_params = results[max_percentage_index][:2]
            step /= 3
            params = [
                optimal_params[0] - step,
                optimal_params[0] + step,
                optimal_params[1] - step,
                optimal_params[1] + step
            ]
            pprint.pprint(results[max_percentage_index])
