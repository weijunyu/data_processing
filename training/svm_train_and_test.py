import subprocess
import pprint
import numpy


def cross_validate(c_low, c_high, gamma_low, gamma_high):
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
                     'hand_5p_scaled_new_features.train'],
                    stdout=subprocess.PIPE
            )

            result_string = str(
                    training_process.stdout.decode('utf-8')
            ).splitlines()
            percentage = float(result_string[-1].split(" ")[-1][:-1])
            results.append([c_power, gamma_power, percentage])
    return results

results = cross_validate(10, 10.5, -5.1, -4.9)
percentage_list = [result[2] for result in results]
max_percentage = max(percentage_list)
max_percentage_index = percentage_list.index(max_percentage)

pprint.pprint(results)
pprint.pprint(results[max_percentage_index])