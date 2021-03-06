import subprocess
import pprint
import numpy
import os


def cross_validate(parameters, log_file):
    c_low = parameters[0]
    c_high = parameters[1]
    gamma_low = parameters[2]
    gamma_high = parameters[3]
    cv_results = []
    for c_p in numpy.linspace(c_low, c_high, 5):
        for gamma_p in numpy.linspace(gamma_low, gamma_high, 5):
            cost_value = 2 ** c_p
            gamma_value = 2 ** gamma_p
            training_process = subprocess.run(
                    ['svm-train.exe',
                     '-v', '10',  # 10 fold cross validation
                     '-c', str(cost_value),  # Cost
                     '-g', str(gamma_value),  # Gamma in kernel function
                     log_file],
                    stdout=subprocess.PIPE
            )

            result_string = str(
                    training_process.stdout.decode('utf-8')
            ).splitlines()
            percentage = float(result_string[-1].split(" ")[-1][:-1])
            cv_results.append([c_p, gamma_p, percentage])
    return cv_results


# Scaling
for file in os.listdir():
    file_name, file_ext = os.path.splitext(file)
    if file_ext == ".unscaled":
        print("scaling " + file)
        subprocess.run(['svm-scale.exe', '-s',
                        'range_' + file_name,
                        file,
                        '>',
                        file_name + ".scaled"],
                       shell=True,  # Necessary for the > operator
                       stdout=subprocess.PIPE)


# Perform grid search on 10-fold cross validation
for file in os.listdir():
    file_name, file_ext = os.path.splitext(file)
    if file_ext == ".scaled":
        pprint.pprint(file)

        step = 18
        lower_bound = -6
        upper_bound = lower_bound + step  # initial bounds are -6 to 12
        params = (lower_bound, upper_bound,
                  lower_bound, upper_bound)
        for i in range(5):
            results = cross_validate(params, file) # Returns a list of [c_power, g_power, cv-percentage] results
            percentage_list = [result[2] for result in results]
            max_percentage = max(percentage_list)
            max_percentage_index = percentage_list.index(max_percentage)
            optimal_params = results[max_percentage_index][:2]
            step /= 2
            params = [
                optimal_params[0] - step,
                optimal_params[0] + step,
                optimal_params[1] - step,
                optimal_params[1] + step
            ]

        [c_power, gamma_power, accuracy] = results[max_percentage_index]
        pprint.pprint(results[max_percentage_index])
        cost = 2 ** c_power
        gamma = 2 ** gamma_power
        subprocess.run(['svm-train.exe',
                        '-c', str(cost),  # Cost
                        '-g', str(gamma),  # Gamma in kernel function
                        '-b', '1',
                        file],
                       stdout=subprocess.PIPE)
